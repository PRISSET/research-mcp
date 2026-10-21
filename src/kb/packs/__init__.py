"""Knowledge packs: curated sources and training recipes for one field, built by GitHub Actions and downloaded in parts.

A pack is a set of records — documents (a trainer's docs page, a README with release notes, an issue thread, a
LoRA card) split into passages, and recipes (the settings a published LoRA was trained with, read from its
safetensors header). The build (`kb pack build`, daily in .github/workflows/packs.yml) compares the new records
with the published ones by id and content hash and publishes only the difference as a numbered change file next
to a periodic snapshot (GitHub release "packs"); manifest.json lists them. A client that is a few versions behind
downloads the missing change files, one that is far behind (or new) the snapshot — whichever is fewer bytes.

Pack records live in the local cache base (kb.sqlite) like anything the agent read, owned by the pack in
`pack_items`: an update touches only the pack's own rows, pruning keeps them, and removing a pack deletes them
unless the notebook cites them.
"""
