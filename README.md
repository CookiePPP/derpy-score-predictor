---

# Derpy Score Predictor

---

Predict the quality of a derpibooru image using it's tags, datetime and score.

The values produced by this network are a representation of image quality that is not affected by the content of the image and time that the image was uploaded.

---

__Examples__

- for a 'safe humanized starlight' image,
getting 100 score would put it in the bottom 15%.
getting 185 score would put it in the bottom 37%.

- for a `explicit, artist:shinodage, shining armor, twilight sparkle, animated, sex" image,
getting 185 score would put it in the bottom 9%,
getting 4009 score would put it in the top 3%.