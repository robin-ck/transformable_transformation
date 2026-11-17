# Delta: viewpoint-futures

## ADDED Requirements

### Requirement: Object-level masking
When the user names an object to change, the system SHALL detect that object in the start image, segment it, and keep the start image unchanged outside a grown and feathered mask of it, so that only the named object (with a small margin) can change. The mask SHALL be stored with the future and shown to the user. If the object is not found, the future SHALL fail with a message naming the object, and SHALL NOT edit the whole image. Masking SHALL only be offered for editors that support it (FLUX.1 Kontext dev).

#### Scenario: Change one object
- **WHEN** the scene's editor is Kontext, the user types "make the chairs red" and "chairs" as the object to change, and clicks Make future
- **THEN** the future shows red chairs, every pixel outside the chairs' mask equals the start image, and a "Changed area" image shows where it changed

#### Scenario: Remove an object
- **WHEN** the user types "remove the bench" with "bench" as the object to change
- **THEN** the mask covers the bench plus a wider margin, so the painted background and the bench's shadow are inside it

#### Scenario: Object not found
- **WHEN** the user names "boat" and no boat is detected in the view
- **THEN** the future fails with 'Nothing matching "boat" was found in this view' before the edit runs, and the gallery does not change

#### Scenario: No object named
- **WHEN** the object field is empty
- **THEN** the future is made as before, without a mask

#### Scenario: Editor without masking
- **WHEN** the scene's editor is SDXL + depth and the user names an object
- **THEN** the future is not started and the message says that only the Kontext editor can limit changes to an object

## MODIFIED Requirements

### Requirement: Future from a viewpoint
The system SHALL let the user pick a viewpoint, optionally a starter direction, write a prompt, optionally name an object to change, and get a restyled image of that viewpoint. The restyle SHALL start from the viewpoint's stored render and depth and SHALL NOT render the scene again. Each future SHALL be stored with its prompt, starter, viewpoint, scene version, editor model, seed, parent (if any), object query (if any) and date, and SHALL appear in the scene's prompt log as a preview.

#### Scenario: Details of a masked future
- **WHEN** the user clicks a future made with "chairs" as the object to change
- **THEN** its details show "Object: chairs (3 found)" and the "Changed area" image
