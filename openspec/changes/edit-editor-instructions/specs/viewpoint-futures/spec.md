# Delta: viewpoint-futures

## ADDED Requirements

### Requirement: Editor instructions
Every text the system adds to a future's or a preview's prompt before the editor gets it (SDXL's negative prompts for futures and for the Preview tool; the object and area instructions for Kontext, and their removal variants), and every rule that rewrites the prompt (the words that make a prompt a removal; the words replaced by the named object), SHALL be visible to the user and changeable per scene. Before a future is made, the system SHALL show exactly what the editor will get. Each future SHALL record the instruction texts it was made with.

#### Scenario: See what is sent
- **WHEN** the scene's editor is Kontext, the user names "bench" as the object and types "make it red"
- **THEN** the line above Make future shows "make the bench red." followed by the object instruction with "the bench" in place of `{target}`

#### Scenario: Change an instruction
- **WHEN** the user changes the area instruction in Settings, saves, paints an area and makes a future
- **THEN** the editor gets the prompt followed by the changed text, and the future's run.json records it

#### Scenario: Empty an instruction
- **WHEN** the user empties the negative prompt and saves
- **THEN** SDXL futures of that scene use no negative prompt

#### Scenario: Reset
- **WHEN** the user clicks Reset instructions to defaults and saves
- **THEN** setup.json stores no changed texts and futures use the defaults

#### Scenario: Bad placeholder
- **WHEN** an instruction uses a placeholder other than `{target}`
- **THEN** the settings are not saved and the message names the allowed placeholder

#### Scenario: Change the removal words
- **WHEN** the user sets the removal words to "remove, weg" and types "bench weg" with "bench" as the object
- **THEN** the future is treated as a removal (removal text, wider margin), and "delete the bench" is not

#### Scenario: Turn off the pronoun rule
- **WHEN** the user empties the replaced words and types "make it red" with "bench" as the object
- **THEN** the editor gets "make it red (the bench)." followed by the object instruction

#### Scenario: Preview tool negative prompt
- **WHEN** the user changes the negative prompt in the Preview sub-tab and clicks Preview this view
- **THEN** SDXL uses that negative prompt, the scene keeps it for later previews, and futures keep their own
