# Delta: viewpoint-futures

## ADDED Requirements

### Requirement: Drawn area (layout)
The user SHALL be able to paint the area of the start image that a future may change. The system SHALL edit a crop of the start image centred on that area and SHALL keep the start image unchanged outside a grown and feathered copy of the area. The area SHALL be stored with the future and shown to the user. An area SHALL NOT be combined with an object to change, and SHALL only be offered for editors that support masking (FLUX.1 Kontext dev).

#### Scenario: Add something new in a place
- **WHEN** the scene's editor is Kontext, the user paints a patch on the meadow, types "a weeping willow" and clicks Make future
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

## MODIFIED Requirements

### Requirement: Future from a viewpoint
The system SHALL let the user pick a viewpoint, optionally a starter direction, write a prompt, optionally name an object to change or paint an area to change, and get a restyled image of that viewpoint. Each future SHALL be stored with its prompt, starter, viewpoint, scene version, editor model, seed, parent (if any), object query or area (if any) and date.

#### Scenario: Details of an area future
- **WHEN** the user clicks a future made with a drawn area
- **THEN** its details show "Area to change: drawn, 4% of the view" and the "Changed area" image
