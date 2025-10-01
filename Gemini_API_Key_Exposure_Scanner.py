#!/usr/bin/env python3

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
try:
    import pyfiglet  # type: ignore
    _HAS_FIGLET = True
except Exception:
    _HAS_FIGLET = False

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
V2_ROOT = os.path.dirname(os.path.abspath(__file__))

# Embedded pricing (snapshot from Gemini_Pricing.html, last updated 2025-09-29)
PRICING: Dict[str, Dict] = {
    "meta": {
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
        "last_updated": "2025-09-29",
        "notes": "Illustrative; prices may vary by region and change over time."
    },
    "pricing": {
        # Text (Gemini 2.5 Flash)
        "text.standard.gemini-2.5-flash": {
            "input_per_million": 0.30,
            "output_per_million": 2.50
        },
        # Imagen 4 (use Fast tier for per-image)
        "image.standard.imagen-4-fast": {
            "per_image": 0.02
        },
        # TTS (Flash Preview)
        "tts.standard.gemini-2.5-flash-preview-tts": {
            "output_per_million_audio": 10.00
        },
        # Video (Veo)
        "video.standard.veo-3-fast": {
            "per_second": 0.15
        },
        "video.standard.veo-3": {
            "per_second": 0.40
        }
    }
}

# ----------------- UI Helpers (ANSI + figlet banner) -----------------

def _supports_color() -> bool:
    return sys.stdout.isatty()

COL = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}

def _c(txt: str, color: str = "", bold: bool = False) -> str:
    if not _supports_color():
        return txt
    parts = []
    if bold:
        parts.append(COL["bold"])
    if color:
        parts.append(COL.get(color, ""))
    parts.append(txt)
    parts.append(COL["reset"])
    return "".join(parts)

BOX_WIDTH = 63
BORDER = "+" + ("\u2500" * BOX_WIDTH) + "+"

def _center_in_box(text_line: str, width: int) -> str:
    if len(text_line) >= width:
        return text_line[:width]
    pad_total = width - len(text_line)
    left = pad_total // 2
    right = pad_total - left
    return (" " * left) + text_line + (" " * right)

def print_banner(name: str, url: str) -> None:
    title = "Gemini API Key Exposure Scanner"
    if _HAS_FIGLET:
        print(BORDER)
        try:
            fig = pyfiglet.figlet_format(title, width=BOX_WIDTH)
            lines = [ln.rstrip("\n") for ln in fig.splitlines() if ln.strip()]
        except Exception:
            lines = [title]
        for line in lines:
            centered = _center_in_box(line, BOX_WIDTH)
            print("| " + _c(centered, "cyan", bold=True) + "|")
        subtitle = f"{name} • {url}"
        print("| " + _c(_center_in_box(subtitle, BOX_WIDTH), "magenta") + "|")
        print(BORDER)
    else:
        print(_c("--- Gemini API Key Exposure Scanner ---", "cyan", bold=True))

def print_header(title: str) -> None:
    print("\n" + _c(title, "cyan", bold=True))

def print_info(msg: str) -> None:
    print(_c("[*] ", "cyan", bold=True) + msg)

def print_ok(msg: str) -> None:
    print(_c("[+] ", "green", bold=True) + msg)

def print_warn(msg: str) -> None:
    print(_c("[!] ", "red", bold=True) + msg)

 

# ----------------- Report Template -----------------
REPORT_TEMPLATE = """# Google API Key Exposed with Gemini-capable API Enabled

- Severity: High
- Date: {{ date_utc }}
- API KEY: {{ api_key }}

## Summary
The provided Google API key is accepted by Gemini endpoints. Detected capabilities:
{{ capabilities_list }}

This enables unauthorized image/video/audio generation and chat, potentially incurring charges and exhausting quotas.

## Steps To Reproduce (PoC)
List available models (capabilities discovery):
```bash
curl -s "{{ base_url }}/models?key={{ api_key }}" | jq '.'
```

Text (chat):
```bash
curl -s -X POST "{{ base_url }}/models/{{ text_model }}:generateContent" \
  -H "x-goog-api-key: {{ api_key }}" -H "Content-Type: application/json" \
  -d '{"contents": [{"parts": [{"text": "Explain how AI works in a few words"}]}]}'
```

Image (Imagen 4 Fast):
```bash
curl -s -X POST "{{ base_url }}/models/imagen-4.0-generate-001:predict" \
  -H "x-goog-api-key: {{ api_key }}" -H "Content-Type: application/json" \
  -d '{
    "instances": [{"prompt": "Robot holding a red skateboard"}],
    "parameters": {"sampleCount": 1}
  }' | jq -r '.. | objects | .bytesBase64Encoded? // empty' | head -n1 | base64 --decode > imagen4.png
```

Video (Veo):
```bash
GEMINI_API_KEY={{ api_key }}
BASE_URL="{{ base_url }}"
operation_name=$(curl -s "$BASE_URL/models/veo-3.0-fast-generate-001:predictLongRunning" \
  -H "x-goog-api-key: $GEMINI_API_KEY" -H "Content-Type: application/json" -X POST \
  -d '{"instances":[{"prompt":"A cinematic 5-second shot of a lantern swaying gently."}]}' | jq -r .name)
while true; do
  status=$(curl -s -H "x-goog-api-key: $GEMINI_API_KEY" "$BASE_URL/$operation_name")
  doneval=$(echo "$status" | jq -r .done)
  if [ "$doneval" = "true" ]; then
    video_uri=$(echo "$status" | jq -r '.response.generateVideoResponse.generatedSamples[0].video.uri')
    curl -L -H "x-goog-api-key: $GEMINI_API_KEY" -o Generated_Video.mp4 "$video_uri"
    break
  fi
  sleep 5
done
```

TTS (single speaker):
```bash
curl -s "{{ base_url }}/models/{{ tts_model }}:generateContent" \
  -H "x-goog-api-key: {{ api_key }}" -H "Content-Type: application/json" \
  -d '{
    "contents":[{"parts":[{"text":"Say cheerfully: Have a wonderful day!"}]}],
    "generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}}
  }' \
  | jq -r '.candidates[0].content.parts[] | select(.inlineData) | .inlineData.data' | head -n1 | base64 --decode > out.pcm

# Convert to WAV or MP3
ffmpeg -y -f s16le -ar 24000 -ac 1 -i out.pcm out.wav
ffmpeg -y -i out.wav out.mp3
```

TTS (multi-speaker):
```bash
curl -s "{{ base_url }}/models/{{ tts_model }}:generateContent" \
  -H "x-goog-api-key: {{ api_key }}" -H "Content-Type: application/json" \
  -d '{
    "contents":[{"parts":[{"text":"Joe: Hows it going today Jane?\nJane: Not too bad, how about you?"}]}],
    "generationConfig":{"responseModalities":["AUDIO"],
      "speechConfig":{"multiSpeakerVoiceConfig":{"speakerVoiceConfigs":[
        {"speaker":"Joe","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}},
        {"speaker":"Jane","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Puck"}}}
      ]}}
    }
  }' \
  | jq -r '.candidates[0].content.parts[] | select(.inlineData) | .inlineData.data' | head -n1 | base64 --decode > out_multi.pcm

# Convert to WAV or MP3
ffmpeg -y -f s16le -ar 24000 -ac 1 -i out_multi.pcm out_multi.wav
ffmpeg -y -i out_multi.wav out_multi.mp3
```

## Evidence
- Saved in: `{{ output_dir }}`
- Files (if generated):
{{ evidence_list }}

## Impact and Pricing (illustrative)
{{ pricing_table }}

Source: Gemini pricing ({{ pricing_last_updated }}): `https://ai.google.dev/gemini-api/docs/pricing`

## Assumptions
- Text ~50 tokens total (input+output); TTS ~2s @ ~32 tokens/sec; Video 5s; Imagen 4 Fast tier.

## Remediation
1. Rotate the compromised API key immediately.
2. Remove keys from client-side code; proxy via server with auth, rate-limits, allowlists.
3. Monitor billing logs for anomalous usage.
"""

def _render_report(ctx: Dict[str, str]) -> str:
    out = REPORT_TEMPLATE
    for k, v in ctx.items():
        out = out.replace("{{ " + k + " }}", v)
    return out

# ----------------- Pricing helpers -----------------

def _estimate_text_cost_assumed() -> float:
    # Assume ~50 total tokens (input+output) for a tiny prompt
    p = PRICING["pricing"]["text.standard.gemini-2.5-flash"]
    total_tokens = 50.0
    # Split 50/50 for simplicity
    cost_in = p["input_per_million"] * ((total_tokens / 2) / 1_000_000.0)
    cost_out = p["output_per_million"] * ((total_tokens / 2) / 1_000_000.0)
    return cost_in + cost_out

def _estimate_imagen4_cost(count: int = 1) -> float:
    return PRICING["pricing"]["image.standard.imagen-4-fast"]["per_image"] * count

def _estimate_tts_cost(seconds: int) -> float:
    # ~32 audio tokens/sec (1,920/min)
    tokens = 32 * seconds
    out_m = PRICING["pricing"]["tts.standard.gemini-2.5-flash-preview-tts"]["output_per_million_audio"]
    return out_m * (tokens / 1_000_000.0)

def _estimate_video_cost(seconds: int, fast: bool = True) -> float:
    key = "video.standard.veo-3-fast" if fast else "video.standard.veo-3"
    per_s = PRICING["pricing"][key]["per_second"]
    return per_s * seconds

# ----------------- HTTP helpers with verbose ----------

def _post(url: str, headers: Dict, json_body: Dict, timeout: int, verbose: bool) -> Tuple[int, Dict]:
    resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
    try:
        data = resp.json()
    except ValueError:
        data = {"_non_json": True, "text": resp.text}
    if verbose:
        print(f"[*] POST {url} -> {resp.status_code}")
    return resp.status_code, data


def _get(url: str, headers: Dict, timeout: int, verbose: bool) -> Tuple[int, Dict]:
    resp = requests.get(url, headers=headers, timeout=timeout)
    try:
        data = resp.json()
    except ValueError:
        data = {"_non_json": True, "text": resp.text}
    if verbose:
        print(f"[*] GET {url} -> {resp.status_code}")
    return resp.status_code, data

# ---------- Extractors ----------

def _extract_inline_audio_data(resp_json: Dict) -> Optional[bytes]:
    try:
        candidates = resp_json.get("candidates", [])
        for cand in candidates:
            content = cand.get("content", {})
            for part in content.get("parts", []):
                if isinstance(part, dict) and "inlineData" in part:
                    data = part["inlineData"].get("data")
                    if data:
                        return base64.b64decode(data)
    except Exception:
        return None
    return None

# Robustly extract base64 image bytes for Imagen 4 predict responses

def _extract_imagen4_image_bytes(resp_json: Dict) -> Optional[bytes]:
    # Common patterns: predictions with bytesBase64Encoded, or nested under objects with this key
    try:
        # direct scan for bytesBase64Encoded anywhere reasonable
        if isinstance(resp_json, dict):
            # Check top-level predictions array
            preds = resp_json.get("predictions") or resp_json.get("prediction")
            if isinstance(preds, list):
                for p in preds:
                    if isinstance(p, dict):
                        b64 = p.get("bytesBase64Encoded")
                        if b64:
                            return base64.b64decode(b64)
                        # sometimes nested
                        for v in p.values():
                            if isinstance(v, dict) and "bytesBase64Encoded" in v:
                                return base64.b64decode(v["bytesBase64Encoded"])
            # Fallback: recursive-ish scan for the key
            stack = [resp_json]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if "bytesBase64Encoded" in cur and isinstance(cur["bytesBase64Encoded"], str):
                        return base64.b64decode(cur["bytesBase64Encoded"])
                    for v in cur.values():
                        if isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(cur, list):
                    for it in cur:
                        if isinstance(it, (dict, list)):
                            stack.append(it)
    except Exception:
        return None
    return None

# ---------- Tests ----------

def _test_text(api_key: str, model: str, verbose: bool) -> bool:
    url = f"{BASE_URL}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": "What's the date today?"}]}]}
    status, j = _post(url, headers, data, timeout=30, verbose=verbose)
    return status == 200 and "candidates" in j

# Imagen 4 image generation (predict)

def _test_imagen4(api_key: str, outdir: str, verbose: bool) -> bool:
    url = f"{BASE_URL}/models/imagen-4.0-generate-001:predict"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    data = {
        "instances": [{"prompt": "A women holding a wine glass having fun at a party."}],
        "parameters": {"sampleCount": 1}
    }
    status, j = _post(url, headers, data, timeout=90, verbose=verbose)
    if status != 200:
        return False
    img_bytes = _extract_imagen4_image_bytes(j)
    if not img_bytes:
        return False
    out_path = os.path.join(outdir, "imagen4.png")
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return True

# TTS single

def _test_tts(api_key: str, model: str, outdir: str, verbose: bool) -> bool:
    url = f"{BASE_URL}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": "Say cheerfully: Have a wonderful day!"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
        },
        "model": model
    }
    status, j = _post(url, headers, data, timeout=60, verbose=verbose)
    if status != 200:
        return False
    audio = _extract_inline_audio_data(j)
    if not audio:
        return False
    pcm_path = os.path.join(outdir, "single_speaker.pcm")
    with open(pcm_path, "wb") as f:
        f.write(audio)
    if _ffmpeg_available():
        wav_path = os.path.join(outdir, "single_speaker.wav")
        if _run_ffmpeg_conversion(pcm_path, wav_path):
            try:
                os.remove(pcm_path)
            except OSError:
                pass
            return True
        print("[!] ffmpeg conversion failed. PCM saved. Convert manually:")
        print(f"    ffmpeg -y -f s16le -ar 24000 -ac 1 -i '{pcm_path}' '{os.path.join(outdir, 'single_speaker.wav')}'")
        return True
    else:
        print("[!] ffmpeg not found. Saved raw PCM. To convert to WAV run:")
        print(f"    ffmpeg -y -f s16le -ar 24000 -ac 1 -i '{pcm_path}' '{os.path.join(outdir, 'single_speaker.wav')}'")
        return True

# TTS multi

def _test_tts_multi(api_key: str, model: str, outdir: str, verbose: bool) -> bool:
    url = f"{BASE_URL}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    convo_text = "Joe: Hows it going today Jane?\nJane: Not too bad, how about you?"
    data = {
        "contents": [{"parts": [{"text": convo_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "multiSpeakerVoiceConfig": {
                    "speakerVoiceConfigs": [
                        {"speaker": "Joe", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
                        {"speaker": "Jane", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}}
                    ]
                }
            }
        },
        "model": model
    }
    status, j = _post(url, headers, data, timeout=120, verbose=verbose)
    if status != 200:
        return False
    audio = _extract_inline_audio_data(j)
    if not audio:
        if verbose:
            print_warn("TTS (multi) returned 200 but no inlineData in parts.")
        return False
    pcm_path = os.path.join(outdir, "multi_speaker.pcm")
    with open(pcm_path, "wb") as f:
        f.write(audio)
    if _ffmpeg_available():
        wav_path = os.path.join(outdir, "multi_speaker.wav")
        if _run_ffmpeg_conversion(pcm_path, wav_path):
            try:
                os.remove(pcm_path)
            except OSError:
                pass
            return True
        print_warn("ffmpeg conversion failed (multi). PCM saved. Convert manually:")
        print(f"    ffmpeg -y -f s16le -ar 24000 -ac 1 -i '{pcm_path}' '{os.path.join(outdir, 'multi_speaker.wav')}'")
        return True
    else:
        print_warn("ffmpeg not found (multi). Saved raw PCM. To convert to WAV run:")
        print(f"    ffmpeg -y -f s16le -ar 24000 -ac 1 -i '{pcm_path}' '{os.path.join(outdir, 'multi_speaker.wav')}'")
        return True

# Video

def _test_video_with_fallback(api_key: str, outdir: str, seconds: int, verbose: bool) -> bool:
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    data = {"instances": [{"prompt": "A group of three men are shown exiting the opening door of an elevator into the lobby of an office building. The men are dressed in suits, dark sunglasses, and earpieces, similar to the agents from the matrix movie series. They walk in unison and in slow motion toward one singular character at the center of the room. The man in the center of the room is wearing a white tank top, a bandolier full of bullets, and is holding a machine gun in either hand while smoking a cigarette. The man in the center of the room looks on at the approaching group of men, grits his teeth, and says \"yippee Kay-yay, BYE BYE...\""}]}
    candidates = [
        "veo-3.0-fast-generate-001",
        "veo-3.0-generate-001",
    ]
    for model in candidates:
        init_url = f"{BASE_URL}/models/{model}:predictLongRunning"
        status, init_json = _post(init_url, headers, data, timeout=30, verbose=verbose)
        if status != 200 or "name" not in init_json:
            continue
        op = init_json["name"]
        if verbose:
            print(f"[*] Video LRO operation: {op}")
        status_url = f"{BASE_URL}/{op}"
        # Poll up to ~6 minutes (36 * 10s)
        for _ in range(36):
            import time
            time.sleep(10)
            s_status, s_json = _get(status_url, headers, timeout=30, verbose=verbose)
            if s_json.get("done"):
                try:
                    uri = s_json["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"]
                except Exception:
                    break
                v = requests.get(uri, headers=headers, timeout=300, allow_redirects=True)
                if v.status_code == 200:
                    out_path = os.path.join(outdir, "Generated_Video.mp4")
                    with open(out_path, "wb") as f:
                        f.write(v.content)
                    return True
                break
    return False

# ---------- Report ----------

def _render_capabilities_list(caps: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    for key in ["text", "image", "video", "tts"]:
        models = caps.get(key, [])
        if models:
            lines.append(f"- {key.capitalize()}: {', '.join(models)}")
    return "\n".join(lines) if lines else "- None detected"


def _pricing_rows(caps: Dict[str, List[str]], pricing_doc: Dict) -> Tuple[str, str]:
    pricing = pricing_doc.get("pricing", {})
    rows: List[str] = []
    
    if pricing.get("image.standard.imagen-4-fast"):
        per_image = pricing["image.standard.imagen-4-fast"].get("per_image")
        rows.append(f"| Image (Imagen 4 Fast) | 1 image | ${per_image:.2f} | Paid per image |")
    if pricing.get("video.standard.veo-3-fast"):
        p = pricing["video.standard.veo-3-fast"].get("per_second")
        rows.append(f"| Video (Veo 3 Fast) | 60 seconds | ${p*60:.2f} | ${p:.2f}/sec |")
    if pricing.get("video.standard.veo-3"):
        p = pricing["video.standard.veo-3"].get("per_second")
        rows.append(f"| Video (Veo 3) | 60 seconds | ${p*60:.2f} | ${p:.2f}/sec |")
    if pricing.get("tts.standard.gemini-2.5-flash-preview-tts"):
        out_m = pricing["tts.standard.gemini-2.5-flash-preview-tts"].get("output_per_million_audio")
        per_min = (out_m / 1_000_000.0) * 1920.0
        rows.append(f"| Audio (TTS Flash Preview) | ~1 minute | ~${per_min:.4f} | ~32 tokens/sec |")
    last = pricing_doc.get("meta", {}).get("last_updated", "unknown")
    table = "\n".join(["| Capability | Unit | Cost | Notes |", "| --- | --- | --- | --- |"] + (rows or ["| — | — | — | — |"]))
    return table, last


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True)
        return True
    except Exception:
        return False


def _run_ffmpeg_conversion(input_pcm: str, output_wav: str) -> bool:
    cmd = ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", input_pcm, output_wav]
    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
        return True
    except Exception:
        return False


def _fetch_models(api_key: str, timeout: int = 30) -> Tuple[int, Dict]:
    url = f"{BASE_URL}/models?key={api_key}"
    resp = requests.get(url, timeout=timeout)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {"error": {"message": "Non-JSON response"}}


def _build_capability_matrix(models_payload: Dict) -> Dict[str, List[str]]:
    models = models_payload.get("models", [])
    names = [m.get("name", "") for m in models]
    methods_by_name = {m.get("name", ""): m.get("supportedGenerationMethods", []) for m in models}

    preferred_text = ["models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5-flash"]
    # Imagen 4 uses predict; capability detection table remains informative for generateContent models
    preferred_tts = ["models/gemini-2.5-flash-preview-tts", "models/gemini-2.5-pro-preview-tts"]
    preferred_video = ["models/veo-3.0-fast-generate-001", "models/veo-3.0-generate-001", "models/veo-3-fast", "models/veo-3"]

    def first_supported(candidates: List[str], require_method: Optional[str] = None) -> Optional[str]:
        for cand in candidates:
            if cand in names:
                if not require_method:
                    return cand
                if require_method in methods_by_name.get(cand, []):
                    return cand
        return None

    text_model = first_supported(preferred_text, require_method="generateContent")
    tts_model = first_supported(preferred_tts, require_method="generateContent")
    video_model = first_supported(preferred_video, require_method=None)

    caps = {"text": [], "image": [], "tts": [], "video": []}
    if text_model:
        caps["text"].append(text_model.replace("models/", ""))
    
    if tts_model:
        caps["tts"].append(tts_model.replace("models/", ""))
    if video_model:
        caps["video"].append(video_model.replace("models/", ""))
    return caps

async def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini API Key Exposure Scanner")
    parser.add_argument("--api-key", dest="api_key", default="", help="Google API key to test.")
    parser.add_argument("--max-cost-usd", type=float, default=0.50, help="Max projected spend for this run (default: 0.50 USD).")
    parser.add_argument("--no-video", action="store_true", help="Skip video test.")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS test.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # Banner
    print_banner("Muhammad Khizer Javed", "whoami.securitybreached.org")

    # Prompt
    if not args.api_key:
        try:
            args.api_key = input(_c("Enter Google API key: ", "yellow", bold=True)).strip()
        except (KeyboardInterrupt, EOFError):
            print_warn("Operation cancelled.")
            return
        if not args.api_key:
            print_warn("API key cannot be empty.")
            return

    out_dir = os.path.join(V2_ROOT, "output", args.api_key)
    os.makedirs(out_dir, exist_ok=True)

    # Discovery
    print_header("Detected capabilities")
    status, payload = _fetch_models(args.api_key)
    if status != 200:
        msg = payload.get("error", {}).get("message", "Failed to list models")
        print_warn(msg)
        return
    caps = _build_capability_matrix(payload)
    # Keep a full capabilities string for the report
    cap_text = _render_capabilities_list(caps)
    # Highlight paid first, then free-tier items
    print_info("Paid highlights:")
    print("- Image: Imagen 4 (Fast)")
    if caps.get("video"):
        print("- Video: Veo 3 Fast")
    print_info("Other supported (Free tier):")
    if caps.get("text"):
        print("- Text: gemini-2.5-flash")
    if caps.get("tts"):
        print("- TTS: gemini-2.5-flash-preview-tts")
    print_info("See report for command to list all available models.")

    # Pricing rows are computed later for the report/output using evidence; no precompute needed here

    print_header("Running tests")

    evidence_files: List[str] = []
    attempted: List[bool] = []
    results: List[bool] = []

    # Text
    if caps.get("text"):
        print_info("Generating text…")
        attempted.append(True)
        ok_text = _test_text(args.api_key, caps["text"][0], args.verbose)
        results.append(ok_text)
        if ok_text:
            print_ok("Text test OK")
        else:
            print_warn("Text test failed")

    # Image (Imagen 4)
    print_info("Generating image (Imagen 4 Fast)…")
    attempted.append(True)
    ok_image = _test_imagen4(args.api_key, out_dir, args.verbose)
    results.append(ok_image)
    if ok_image:
        print_ok("Image saved: imagen4.png")
        evidence_files.append("- imagen4.png")
    else:
        print_warn("Imagen 4 test failed or not supported")

    # TTS single
    if caps.get("tts") and not args.no_tts:
        print_info("Generating audio (single)…")
        attempted.append(True)
        ok_tts = _test_tts(args.api_key, caps["tts"][0], out_dir, args.verbose)
        results.append(ok_tts)
        if ok_tts:
            if os.path.exists(os.path.join(out_dir, "single_speaker.wav")):
                evidence_files.append("- single_speaker.wav")
            elif os.path.exists(os.path.join(out_dir, "single_speaker.pcm")):
                evidence_files.append("- single_speaker.pcm")
        else:
            print_warn("TTS test failed or not supported")

    # TTS multi
    if caps.get("tts") and not args.no_tts:
        print_info("Generating audio (multi-speaker)…")
        attempted.append(True)
        ok_tts_multi = _test_tts_multi(args.api_key, caps["tts"][0], out_dir, args.verbose)
        results.append(ok_tts_multi)
        if ok_tts_multi:
            if os.path.exists(os.path.join(out_dir, "multi_speaker.wav")):
                evidence_files.append("- multi_speaker.wav")
            elif os.path.exists(os.path.join(out_dir, "multi_speaker.pcm")):
                evidence_files.append("- multi_speaker.pcm")
        else:
            print_warn("TTS (multi) test failed or not supported")

    # Video (always attempt unless disabled)
    if not args.no_video:
        print_info("Generating video (Veo Fast fallback, 5s)…")
        attempted.append(True)
        ok_video = _test_video_with_fallback(args.api_key, out_dir, seconds=5, verbose=args.verbose)
        results.append(ok_video)
        if ok_video:
            print_ok("Video saved: Generated_Video.mp4")
            evidence_files.append("- Generated_Video.mp4")
        else:
            print_warn("Video test failed or not supported")

    # Results summary
    print_header("Results")
    if caps.get("text") and attempted:
        # first attempted corresponds to text when present
        print_ok("Text") if results[0] else None
    for f in evidence_files:
        print_ok(f.replace("- ", ""))

    # Impact estimates (illustrative)
    print_header("Impact estimate (illustrative)")
    total_cost = 0.0
    rows: List[Dict[str, str]] = []

    # Text (assumed tiny prompt)
    text_cost = _estimate_text_cost_assumed()
    rows.append({"Capability": "Text", "Unit": "~50 tokens", "Cost": f"~${text_cost:.5f}", "Notes": "Gemini 2.5 Flash (after free tier)"})
    total_cost += text_cost

    # Image (Imagen 4)
    if any(x.endswith("imagen4.png") for x in (p.replace("- ", "") for p in evidence_files)):
        c = _estimate_imagen4_cost(count=1)
        total_cost += c
        rows.append({"Capability": "Image", "Unit": "1 image", "Cost": f"${c:.2f}", "Notes": "Imagen 4 Fast (paid)"})

    # TTS single (~2s)
    if any(x.endswith("single_speaker.wav") or x.endswith("single_speaker.pcm") for x in (p.replace("- ", "") for p in evidence_files)):
        c = _estimate_tts_cost(seconds=2)
        total_cost += c
        rows.append({"Capability": "Audio (single)", "Unit": "~2 seconds", "Cost": f"~${c:.5f}", "Notes": "~32 tokens/sec (after free tier)"})

    # TTS multi (~2s)
    if any(x.endswith("multi_speaker.wav") or x.endswith("multi_speaker.pcm") for x in (p.replace("- ", "") for p in evidence_files)):
        c = _estimate_tts_cost(seconds=2)
        total_cost += c
        rows.append({"Capability": "Audio (multi)", "Unit": "~2 seconds", "Cost": f"~${c:.5f}", "Notes": "~32 tokens/sec (after free tier)"})

    # Video (5s fast)
    if any(x.endswith("Generated_Video.mp4") for x in (p.replace("- ", "") for p in evidence_files)):
        c = _estimate_video_cost(seconds=5, fast=True)
        total_cost += c
        rows.append({"Capability": "Video", "Unit": "5 seconds", "Cost": f"${c:.2f}", "Notes": "Veo 3 Fast (paid)"})

    
    headers = ["Capability", "Unit", "Cost", "Notes"]
    col_widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(r[h]))
    def fmt_row(r: Dict[str, str]) -> str:
        return " | ".join(r[h].ljust(col_widths[h]) for h in headers)
    header_line = fmt_row({h: h for h in headers})
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    if rows:
        print(header_line)
        print(sep_line)
        for r in rows:
            print(fmt_row(r))
        print_ok(f"Estimated total (run): ${total_cost:.2f}")
        print(f"Source: Gemini pricing ({PRICING['meta']['last_updated']}): https://ai.google.dev/gemini-api/docs/pricing")
    else:
        print_warn("No priced capabilities were generated.")

    
    print_header("Report")
    md_header = "| Capability | Unit | Cost | Notes |"
    md_sep = "| --- | --- | --- | --- |"
    if rows:
        md_rows = ["| " + r["Capability"] + " | " + r["Unit"] + " | " + r["Cost"] + " | " + r["Notes"] + " |" for r in rows]
    else:
        md_rows = ["| — | — | — | — |"]
    pricing_table_md = "\n".join([md_header, md_sep] + md_rows)

    ctx = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "api_key": args.api_key,
        "capabilities_list": cap_text,
        "base_url": BASE_URL,
        "text_model": (caps.get("text") or [""])[0],
        "image_model": "imagen-4.0-generate-001",
        "tts_model": (caps.get("tts") or [""])[0],
        "output_dir": out_dir,
        "evidence_list": "\n".join(evidence_files) if evidence_files else "- (none)",
        "pricing_table": pricing_table_md,
        "pricing_last_updated": PRICING["meta"]["last_updated"],
    }
    report = _render_report(ctx)
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print_ok(f"Report written: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
