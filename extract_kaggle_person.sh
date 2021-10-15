#!/bin/bash

in_dir="img/faces/person"
out="train_images_kaggle_person"
mkdir -p "$out" "$out/0" "$out/1"

crop() {
  img="$1"
  local n="$(basename "${img%%.*}")"
  convert "$img" \
    -resize 72x72 -crop 36x36+18+18 \
    -set colorspace Gray -separate -average \
    "$out/1/$n.pgm"
}

for img in "$in_dir"/*; do
  [ -e "$img" ] || continue

  echo "$img"
  crop "$img" &
done
