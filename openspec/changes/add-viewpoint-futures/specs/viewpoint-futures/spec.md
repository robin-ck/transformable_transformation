# Delta: viewpoint-futures

## ADDED Requirements

### Requirement: Saved viewpoints
The system SHALL let the user save the viewer's current camera as a named viewpoint of the scene version shown, and rename, reorder and remove viewpoints. Saving a viewpoint SHALL render its colour image and depth once and store them, and SHALL show that render as the viewpoint's thumbnail. Picking a viewpoint SHALL move the viewer to its camera. Removing a viewpoint SHALL NOT delete its futures.

#### Scenario: Save a viewpoint
- **WHEN** the user has moved the viewer to look along the river from the bridge on v1 and saves it as "bridge"
- **THEN** "bridge" appears in the scene's viewpoint list with a thumbnail of its render, pinned to v1

#### Scenario: Go to a viewpoint
- **WHEN** the user picks "east bank" in the list
- **THEN** the viewer shows the scene from the east bank camera

#### Scenario: New scene version
- **WHEN** a v2 of the scene is created after "bridge" was saved on v1
- **THEN** "bridge" still shows v1 and its futures are unchanged

#### Scenario: Remove a viewpoint
- **WHEN** the user removes "east bank", which has futures
- **THEN** it no longer appears in the list, and its futures stay in the dataset

### Requirement: Future from a viewpoint
The system SHALL let the user pick a viewpoint, optionally a starter direction, write a prompt and get a restyled image of that viewpoint. The restyle SHALL start from the viewpoint's stored render and depth and SHALL NOT render the scene again. Each future SHALL be stored with its prompt, starter, viewpoint, scene version, editor model, seed, parent (if any) and date, and SHALL appear in the scene's prompt log as a preview.

#### Scenario: Make a future
- **WHEN** the user picks "bridge", chooses the starter "nature first", types "willows and a reed bank where the parking is" and clicks Make
- **THEN** a restyled image of the bridge view is shown, and the gallery of "bridge" lists it with the full prompt

#### Scenario: Same start every time
- **WHEN** the user makes two futures of "bridge" with different prompts
- **THEN** both restyles start from the same stored render

#### Scenario: Future from the current view
- **WHEN** the user moves the viewer to the old weir, which is not a saved viewpoint, and clicks Make with "a fish ladder next to the weir"
- **THEN** the current camera is saved as a new viewpoint, the future is made from it in the same job, and the gallery of that viewpoint lists it

#### Scenario: Failed future
- **WHEN** the restyle job fails
- **THEN** an error message is shown, the user can try again, and the gallery does not show the failed future

### Requirement: Build on a future
The system SHALL let the user start a new future from an existing one: the new future restyles the existing future's image with the new prompt, keeps the viewpoint's depth, and records the existing future as its parent.

#### Scenario: Start from a future
- **WHEN** the user opens a future and chooses "Start from this" with the prompt "add a footbridge over the reeds"
- **THEN** the new image builds on the chosen image, and its details name the chosen future as the parent

### Requirement: Compare futures
The system SHALL show all futures of one viewpoint in a gallery, with the plain render as the first tile and then the futures newest first, and SHALL show any two of them (or one and the plain render) side by side at the same size with their prompts.

#### Scenario: Gallery of a viewpoint
- **WHEN** the user selects "bridge"
- **THEN** a grid shows the plain render followed by every future of "bridge", each with the start of its prompt

#### Scenario: Side by side
- **WHEN** the user selects two futures in the gallery and chooses Compare
- **THEN** both images are shown next to each other at the same size, each with its full prompt, starter and date

#### Scenario: Futures from earlier sessions
- **WHEN** the scene has futures made on an earlier day
- **THEN** they are listed in the same gallery with their dates

### Requirement: Starter prompts
The system SHALL let the user define starter directions per scene, each with a label and prompt text. The prompt text of a chosen starter SHALL be put before the user's own text, and both SHALL be stored with the future.

#### Scenario: Starter only
- **WHEN** the user chooses "flood safe" and leaves their own text empty
- **THEN** the future is made from the starter's text alone

#### Scenario: Edit starters
- **WHEN** the user changes the text of "housing"
- **THEN** new futures use the new text, and earlier futures keep the text they were made with

### Requirement: Exchangeable image editor
The system SHALL restyle previews and futures through one editor interface, with SDXL + depth ControlNet as the default editor, and SHALL store the editor used with every future. The user SHALL be able to choose the editor per scene among the registered ones. Adding another editor SHALL NOT change how earlier futures are stored or shown.

#### Scenario: Default editor
- **WHEN** the user makes a future on a scene with no editor chosen
- **THEN** it is made with SDXL + depth ControlNet, and its details name that model

#### Scenario: Editor changed later
- **WHEN** a second editor is registered and selected for the scene
- **THEN** new futures name the new editor, and earlier futures still show and name SDXL + depth ControlNet

### Requirement: Future turnaround
The system SHALL show a future within 20 seconds of clicking Make when a warm preview container and a free GPU are available, and within 90 seconds with a cold container, not counting time spent waiting for a free GPU.

#### Scenario: Futures back to back
- **WHEN** the user makes a future within 5 minutes after the previous one finished
- **THEN** the image is shown within 20 seconds of clicking Make
