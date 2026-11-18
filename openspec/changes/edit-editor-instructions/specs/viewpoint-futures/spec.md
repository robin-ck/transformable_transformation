# Delta: viewpoint-futures

## ADDED Requirements

### Requirement: Editor instructions
Every text the system adds to a future's prompt before the editor gets it (SDXL's negative prompt; the object and area instructions for Kontext, and their removal variants) SHALL be visible to the user and changeable per scene. Before a future is made, the system SHALL show exactly what the editor will get. Each future SHALL record the instruction texts it was made with.

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
