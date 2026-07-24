# Fonts — Amazon Ember (self-hosted)

The UI uses **Amazon Ember** (Alexa/Echo brand typeface) per `styleguide.md` / FR-13.
These are **proprietary Amazon fonts**. The `@font-face` rules in `src/index.css`
reference the TrueType (`.ttf`) files below by their exact names; if a file is absent
the UI falls back to the system font stack (`font-display: swap`), so the app still
works — it just won't render in Amazon Ember.

Vite serves `public/` at the site root, so these resolve at `/fonts/<name>.ttf`.

## Files used (mapped in `src/index.css`)

| File | Weight | Style |
|------|--------|-------|
| `AmazonEmber_Th.ttf` | 100–200 (Thin) | normal |
| `AmazonEmber_ThIt.ttf` | 100–200 (Thin) | italic *(not currently mapped)* |
| `AmazonEmber_Lt.ttf` | 300 (Light) | normal |
| `AmazonEmber_LtIt.ttf` | 300 (Light) | italic |
| `AmazonEmber_Rg.ttf` | 400 (Regular) | normal |
| `AmazonEmber_RgIt.ttf` | 400 (Regular) | italic |
| `Amazon-Ember-Medium.ttf` | 500 (Medium) | normal |
| `Amazon-Ember-MediumItalic.ttf` | 500 (Medium) | italic |
| `AmazonEmber_Bd.ttf` | 700 (Bold) | normal |
| `AmazonEmber_BdIt.ttf` | 700 (Bold) | italic |
| `AmazonEmber_He.ttf` | 800–900 (Heavy) | normal |
| `AmazonEmber_HeIt.ttf` | 800–900 (Heavy) | italic |

There is no separate "Amazon Ember Display" cut in this set — headings (`--font-display`)
use **Amazon Ember** at weight 700/800.

## Where to get them
Source the fonts from the **Amazon Type Library / brand portal** (Amazon's in-house
typeface, usable across Amazon products).

**Licensing note:** this is a public AWS Guidance repo and the UI serves these files to
end users' browsers (redistribution). Only include the font here if that is
license-permitted for your distribution of this repo.
