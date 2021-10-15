#!/bin/bash

in_dir="lfw-deepfunneled"
out="train_images_lfw-deepfunneled"
mkdir -p "$out" "$out/0" "$out/1"

crop() {
  img="$1"
  local n="$(basename "${img%%.*}")"
  convert "$img" \
    -resize 72x72 -crop 36x36+18+18 \
    -set colorspace Gray -separate -average \
    "$out/1/$n.pgm"
  convert "$img" \
    -resize 252x252 \
    -set colorspace Gray -separate -average \
    -crop 36x36 \
    "$out/0/${n}_%d.pgm"
}

for person_dir in "$in_dir"/*; do
  [ -e "$person_dir" ] || continue
  for img in "$person_dir"/*; do
    [ -e "$img" ] || continue

    echo "$img"
    crop "$img" &
  done
done
