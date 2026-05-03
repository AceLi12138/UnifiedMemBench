# Event-Source Release Artifacts

This directory contains small event-source artifacts used by the current
UnifiedMemBench release candidate.

Files:

- `personas_1000_v3.json`: synthetic evolving persona records.
- `stories_v4.json`: event-centric synthetic story source with timestamped events.
- `stories_v4_characters_qa.json`: six-family QA tasks generated from the event source.
- `stories_v4_validation_report.json`: validation summary for the event source.

The initial MBTI-style seed bank used to create the persona source is derived
from the CharacterChat MBTI-1024 Bank:

https://github.com/morecry/CharacterChat

The raw external seed bank is not redistributed in this repository. To
regenerate the source, provide your own compatible seed bank file to
`data_gen/generate_evolving_personas.py`.

