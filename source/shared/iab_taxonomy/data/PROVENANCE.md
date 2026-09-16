# Bundled IAB Tech Lab taxonomy data

Unmodified copies of two IAB Tech Lab published taxonomy files. Only the filenames were changed, to
avoid spaces in paths; the file contents are byte-identical to what IAB published, which is what the
checksums below verify.

## Files

| Bundled as | Published as | Version | Rows | SHA-256 |
|---|---|---|---|---|
| `content_taxonomy_3_1.tsv` | `Content Taxonomy 3.1.tsv` | Content Taxonomy 3.1 | 705 | `7212cdc496ba347a03e703b1932bdcdd4fd29089b058f4edeb4d3da1f1222ea7` |
| `audience_taxonomy_1_1.tsv` | `Audience Taxonomy 1.1.tsv` | Audience Taxonomy 1.1 | 1558 | `0216547402f3dc028f5ec1bb78c648eed68a81ea3b3b94862e6f6caa9db3ad3b` |

**Source**: <https://github.com/InteractiveAdvertisingBureau/Taxonomies>
- `Content Taxonomies/Content Taxonomy 3.1.tsv`
- `Audience Taxonomies/Audience Taxonomy 1.1.tsv`

**Retrieved**: 2026-09-08

## Verifying

```bash
shasum -a 256 content_taxonomy_3_1.tsv audience_taxonomy_1_1.tsv
```

The values must match the table above. They were verified at the time of bundling.

## Why these versions

**Content Taxonomy 3.1** is the current content taxonomy, `cattax` value 9 in
[AdCOM's Category Taxonomies list](https://github.com/InteractiveAdvertisingBureau/AdCOM/blob/main/AdCOM%20v1.0%20FINAL.md).
Content Taxonomy 1.0 and 2.0 are deprecated by IAB because they lack the Special Category Data
extension. Content 3.x is a documented breaking change from 1.0 through 2.2, so the versions cannot be
merged.

**Audience Taxonomy 1.1** is `cattax` value 4. Its Tier 1 divides segments into Demographic, Interest
and Purchase Intent.

Pinned by filename and checksum, and never fetched at build time. IAB versions these independently, so
a build that fetched the latest could change segment output with no code change.

## Licence and attribution

Both files are published by IAB Technology Laboratory under the
[Creative Commons Attribution 3.0 licence](https://creativecommons.org/licenses/by/3.0/).

> IAB Tech Lab Content Taxonomy and Audience Taxonomy, copyright IAB Technology Laboratory, licensed
> under CC BY 3.0. Source: <https://github.com/InteractiveAdvertisingBureau/Taxonomies>

## What is not bundled, and why

**Content Taxonomy 2.x.** Its identifiers are not interchangeable with 3.x, and supporting it would mean
a third table and a second mapping for a taxonomy IAB has moved past. A request declaring `cattax` 2, 5
or 6 is treated as an unrecognised taxonomy.

**Ad Product Taxonomy.** Describes the product being advertised, not the page or the person.

## Note on the mapping between them

IAB publishes mappings between several of its taxonomies, but **not** between Content and Audience. The
`Taxonomy Mappings` folder contains Content 1.0 to Ad Product 2.0, Content 1.0 to Content 2.0, Content
2.0 to 2.1, Content 2.1 to Ad Product 2.0, Ad Product reverse mappings, and CTV and podcast genre
mappings. There is no Content-to-Audience file.

The mapping this project uses is therefore **ours, not IAB's**. It joins a content category's tier path
against Audience Taxonomy Interest tier paths, which works because the two taxonomies share tier
vocabulary by design. It resolves 565 of the 705 Content 3.1 categories. See
