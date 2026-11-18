# Delta: viewpoint-futures

## ADDED Requirements

### Requirement: Try several editors
The user SHALL be able to tick several editors and make one future with each of them from the same start image, prompt, object or area, and seed. The system SHALL run these futures in parallel, one job per editor. Each future SHALL record and show its editor, and each gallery tile SHALL name its editor.

#### Scenario: Same prompt, four editors
- **WHEN** the user ticks Kontext, Qwen-Image-Edit, FLUX.2 klein and Step1X-Edit, types "remove the bench" with "bench" as the object to change, and clicks Make future
- **THEN** four futures start with the same seed, and each appears in the gallery when it finishes, with its editor's name on its tile

#### Scenario: Restyle editor ticked with an object
- **WHEN** an object or area is given and a restyle editor (SDXL + depth or SDXL + union) is also ticked
- **THEN** no future is started for that editor, and the status says it cannot limit changes to an object or area

#### Scenario: Compare editors
- **WHEN** the user clicks two of these futures and Compare
- **THEN** they are shown side by side, each captioned with its editor

## MODIFIED Requirements

### Requirement: Exchangeable image editor
The system SHALL restyle previews and futures through one editor interface, with SDXL + depth ControlNet as the default editor, and SHALL store the editor used with every future. The user SHALL be able to choose the editor per scene among the registered ones: SDXL + depth ControlNet, SDXL + union ControlNet (Lightning), FLUX.1 Kontext dev (research use only), Qwen-Image-Edit-2511 (Lightning), FLUX.2 klein 4B and Step1X-Edit v1.2. Every registered editor SHALL use models whose licences are in the project's allowed list, or in the research-only list for an editor that is never the default. Adding another editor SHALL NOT change how earlier futures are stored or shown.

#### Scenario: Default editor
- **WHEN** the user makes a future on a scene with no editor chosen
- **THEN** it is made with SDXL + depth ControlNet, and its details name that model

#### Scenario: Editor changed later
- **WHEN** a second editor is registered and selected for the scene
- **THEN** new futures name the new editor, and earlier futures still show and name SDXL + depth ControlNet

#### Scenario: New open-licence editor
- **WHEN** the scene's editor is Qwen-Image-Edit-2511 and the user types "make the bridge wooden" and clicks Make future
- **THEN** the future is made by Qwen-Image-Edit-2511 with its Lightning LoRA, and its details name both models and their pinned revisions

#### Scenario: Step1X rewrites the prompt
- **WHEN** a future is made with Step1X-Edit
- **THEN** its details show both the prompt sent and the prompt as Step1X rewrote it

### Requirement: Object-level masking
When the user names an object to change, the system SHALL detect that object in the start image, segment it, and keep the start image unchanged outside a grown and feathered mask of it, so that only the named object (with a small margin) can change. The mask SHALL be stored with the future and shown to the user. The system SHALL also store and show a labelled image of the start image with every detection's box and mask, each numbered and labelled with its matched phrase and score in its own colour, and SHALL list the detections in the future's details. If the object is not found, the future SHALL fail with a message naming the object, and SHALL NOT edit the whole image. Masking SHALL be offered for every instruction editor (FLUX.1 Kontext dev, Qwen-Image-Edit-2511, FLUX.2 klein 4B, Step1X-Edit v1.2) and SHALL NOT be offered for restyle editors (SDXL + depth, SDXL + union).

#### Scenario: See what was detected
- **WHEN** a future with "chairs" as the object to change finishes and three chairs and a stool were matched
- **THEN** a labelled image shows four coloured boxes and masks numbered 1 to 4 with label and score, captioned "Detected: 3 × chair, 1 × stool", and the details list each detection

#### Scenario: Gallery and Compare
- **WHEN** the user clicks that future in the gallery, or compares it with another
- **THEN** its labelled detection image and detection list are shown with it

#### Scenario: Detect before making a future
- **WHEN** the scene's editor is an instruction editor, a saved view is selected, the user types "chairs" in "Object to change" and clicks Detect objects
- **THEN** the labelled image of the view's plain render and its caption are shown, nothing is stored, and no future is made; if nothing matches, the message names the object

#### Scenario: Older masked future
- **WHEN** the user clicks a masked future made before add-detection-preview
- **THEN** its details list the detections from `run.json` and no labelled image is shown

#### Scenario: Masking with a new editor
- **WHEN** the scene's editor is FLUX.2 klein, the user types "make the chairs red" with "chairs" as the object to change
- **THEN** only the chairs' mask can change, as with Kontext

#### Scenario: Restyle editor with an object
- **WHEN** the scene's editor is SDXL + union ControlNet and the user names an object
- **THEN** the future is not started and the message says that only instruction editors can limit changes to an object

### Requirement: Drawn area (layout)
The user SHALL be able to paint the area of the start image that a future may change. The system SHALL edit a crop of the start image centred on that area and SHALL keep the start image unchanged outside a grown and feathered copy of the area. The area SHALL be stored with the future and shown to the user. An area SHALL NOT be combined with an object to change. Areas SHALL be offered for every instruction editor (FLUX.1 Kontext dev, Qwen-Image-Edit-2511, FLUX.2 klein 4B, Step1X-Edit v1.2) and SHALL NOT be offered for restyle editors.

#### Scenario: Add something new in a place
- **WHEN** the scene's editor is an instruction editor, the user paints a patch on the meadow, types "a weeping willow" and clicks Make future
- **THEN** a willow appears inside the painted patch, every pixel outside the grown area equals the start image, and a "Changed area" image shows where it changed

#### Scenario: Change an area without clear edges
- **WHEN** the user paints the river bank and types "a reed bed with shallow water"
- **THEN** only the painted bank changes

#### Scenario: Empty or tiny area
- **WHEN** the painted area covers less than 0.2 % of the view
- **THEN** the future is not started and the message asks for a larger area or an empty drawing

#### Scenario: Area and object together
- **WHEN** the user paints an area and also names an object to change
- **THEN** the future is not started and the message says to use one of them

#### Scenario: Start image changed after drawing
- **WHEN** the user paints an area on one view and then picks another view or "Start from this"
- **THEN** the editor shows the new start image with no area drawn
