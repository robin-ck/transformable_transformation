# Delta: viewpoint-futures

## MODIFIED Requirements

### Requirement: Object-level masking
When the user names an object to change, the system SHALL detect that object in the start image, segment it, and keep the start image unchanged outside a grown and feathered mask of it, so that only the named object (with a small margin) can change. The mask SHALL be stored with the future and shown to the user. The system SHALL also store and show a labelled image of the start image with every detection's box and mask, each numbered and labelled with its matched phrase and score in its own colour, and SHALL list the detections in the future's details. If the object is not found, the future SHALL fail with a message naming the object, and SHALL NOT edit the whole image. Masking SHALL only be offered for editors that support it (FLUX.1 Kontext dev).

#### Scenario: See what was detected
- **WHEN** a future with "chairs" as the object to change finishes and three chairs and a stool were matched
- **THEN** a labelled image shows four coloured boxes and masks numbered 1 to 4 with label and score, captioned "Detected: 3 × chair, 1 × stool", and the details list each detection

#### Scenario: Gallery and Compare
- **WHEN** the user clicks that future in the gallery, or compares it with another
- **THEN** its labelled detection image and detection list are shown with it

#### Scenario: Detect before making a future
- **WHEN** the scene's editor is Kontext, a saved view is selected, the user types "chairs" in "Object to change" and clicks Detect objects
- **THEN** the labelled image of the view's plain render and its caption are shown, nothing is stored, and no future is made; if nothing matches, the message names the object

#### Scenario: Older masked future
- **WHEN** the user clicks a masked future made before this change
- **THEN** its details list the detections from `run.json` and no labelled image is shown
