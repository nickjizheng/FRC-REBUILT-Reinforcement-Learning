#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 INPUT_AVI OUTPUT_PREFIX" >&2
  exit 2
fi

input="$1"
prefix="$2"
ffmpeg="${FFMPEG_BIN:-/root/autodl-tmp/public_video_tools/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
master="${prefix}-master.mp4"
preview="${prefix}-preview.mp4"
poster="${prefix}-poster.png"
expected_frames="${EXPECTED_FRAMES:-1600}"
expected_fps="${EXPECTED_FPS:-10}"
expected_width="${EXPECTED_WIDTH:-1640}"
expected_height="${EXPECTED_HEIGHT:-720}"
expected_duration_us="${EXPECTED_DURATION_US:-160000000}"
expected_input_codec="${EXPECTED_INPUT_CODEC:-mjpeg}"
master_preset="${MASTER_PRESET:-slow}"
preview_preset="${PREVIEW_PRESET:-slow}"

die() {
  echo "$*" >&2
  exit 1
}

for value in "$expected_frames" "$expected_fps" "$expected_width" \
    "$expected_height" "$expected_duration_us"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "invalid expected media value: $value"
done
[[ -x "$ffmpeg" ]] || die "ffmpeg is not executable: $ffmpeg"
[[ -f "$input" ]] || die "missing input: $input"
[[ "$master_preset" =~ ^(ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow)$ ]] || \
  die "invalid master preset: $master_preset"
[[ "$preview_preset" =~ ^(ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow)$ ]] || \
  die "invalid preview preset: $preview_preset"

media_metadata() {
  local path="$1"
  # Input-only ffmpeg intentionally returns nonzero after printing valid stream
  # metadata, so capture its output without exposing that status to pipefail.
  "$ffmpeg" -nostdin -hide_banner -i "$path" 2>&1 || true
}

verify_video() {
  local path="$1"
  local role="$2"
  local codec="$3"
  local metadata stream progress frames out_time_us

  [[ -s "$path" ]] || die "$role is missing or empty: $path"
  metadata="$(media_metadata "$path")"
  stream="$(printf '%s\n' "$metadata" | grep -m1 -E 'Stream .*Video:' || true)"
  [[ -n "$stream" ]] || die "$role has no readable video stream: $path"
  printf '%s\n' "$stream" | grep -Eq "Video: ${codec}([, (]|$)" || \
    die "$role codec mismatch: $stream"
  printf '%s\n' "$stream" | grep -Eq "(^|[^0-9])${expected_width}x${expected_height}([^0-9]|$)" || \
    die "$role resolution mismatch: $stream"
  printf '%s\n' "$stream" | grep -Eq "(^|[ ,])${expected_fps}(\\.0+)? fps([, ]|$)" || \
    die "$role frame-rate mismatch: $stream"
  if [[ "$role" != "intermediate" ]]; then
    printf '%s\n' "$stream" | grep -q 'yuv420p' || \
      die "$role pixel-format mismatch: $stream"
  fi
  if printf '%s\n' "$metadata" | grep -Eq 'Stream .*Audio:'; then
    die "$role unexpectedly contains audio: $path"
  fi

  if ! progress="$("$ffmpeg" -nostdin -hide_banner -v error -i "$path" \
      -map 0:v:0 -an -sn -dn -f null - -progress pipe:1)"; then
    die "$role failed a full decode: $path"
  fi
  frames="$(printf '%s\n' "$progress" | awk -F= '$1 == "frame" { value=$2 } END { print value }')"
  out_time_us="$(printf '%s\n' "$progress" | awk -F= '$1 == "out_time_us" { value=$2 } END { print value }')"
  [[ "$frames" == "$expected_frames" ]] || \
    die "$role decoded frame count mismatch: ${frames:-missing}/$expected_frames"
  [[ "$out_time_us" == "$expected_duration_us" ]] || \
    die "$role decoded duration mismatch: ${out_time_us:-missing}/$expected_duration_us us"
  printf 'MEDIA_VERIFY_OK role=%s frames=%s duration_us=%s path=%s\n' \
    "$role" "$frames" "$out_time_us" "$path"
}

verify_poster() {
  local metadata stream
  [[ -s "$poster" ]] || die "poster is missing or empty: $poster"
  metadata="$(media_metadata "$poster")"
  stream="$(printf '%s\n' "$metadata" | grep -m1 -E 'Stream .*Video:' || true)"
  [[ -n "$stream" ]] || die "poster has no readable image stream: $poster"
  printf '%s\n' "$stream" | grep -Eq 'Video: png([, (]|$)' || \
    die "poster codec mismatch: $stream"
  printf '%s\n' "$stream" | grep -Eq "(^|[^0-9])${expected_width}x${expected_height}([^0-9]|$)" || \
    die "poster resolution mismatch: $stream"
  printf 'MEDIA_VERIFY_OK role=poster frames=1 path=%s\n' "$poster"
}

verify_video "$input" intermediate "$expected_input_codec"

existing=0
for output in "$master" "$preview" "$poster"; do
  [[ -e "$output" ]] && existing=$((existing + 1))
done

if [[ "$existing" -eq 0 ]]; then
  master_tmp="${prefix}.master.$$.partial.mp4"
  preview_tmp="${prefix}.preview.$$.partial.mp4"
  poster_tmp="${prefix}.poster.$$.partial.png"
  trap 'rm -f -- "$master_tmp" "$preview_tmp" "$poster_tmp"' EXIT

  "$ffmpeg" -nostdin -hide_banner -loglevel error -i "$input" -an \
    -c:v libx264 -preset "$master_preset" -crf 17 -tune animation \
    -pix_fmt yuv420p -movflags +faststart "$master_tmp"

  "$ffmpeg" -nostdin -hide_banner -loglevel error -i "$input" -an \
    -c:v libx264 -preset "$preview_preset" -b:v 420k -maxrate 560k -bufsize 1120k \
    -tune animation -pix_fmt yuv420p -movflags +faststart "$preview_tmp"

  "$ffmpeg" -nostdin -hide_banner -loglevel error -ss 00:01:20 -i "$master_tmp" \
    -frames:v 1 "$poster_tmp"

  mv -- "$master_tmp" "$master"
  mv -- "$preview_tmp" "$preview"
  mv -- "$poster_tmp" "$poster"
  trap - EXIT
elif [[ "$existing" -eq 3 ]]; then
  for output in "$master" "$preview" "$poster"; do
    [[ "$output" -nt "$input" ]] || \
      die "refusing to reuse output that is not newer than input: $output"
  done
  echo "ENCODE_REUSE_PENDING_VALIDATION prefix=$prefix"
else
  die "partial output set exists for prefix: $prefix ($existing/3); refusing recovery"
fi

verify_video "$master" master h264
verify_video "$preview" preview h264
verify_poster
if [[ "$existing" -eq 3 ]]; then
  echo "ENCODE_REUSE_VALIDATED prefix=$prefix"
fi
sha256sum "$input" "$master" "$preview" "$poster"
stat -c '%n %s bytes' "$input" "$master" "$preview" "$poster"
