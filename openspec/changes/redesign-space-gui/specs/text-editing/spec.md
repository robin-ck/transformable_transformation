# Delta: text-editing

## MODIFIED Requirements

### Requirement: Add object from text
The system SHALL generate a 3D object from a text prompt, show it in the viewer, let the user move, rotate and scale it, and on confirmation merge it into a new scene version. The user SHALL be able to place the object by clicking a point in the scene and to move it by dragging it in the viewer; sliders SHALL remain available and SHALL show the same position, turn and size as the viewer. The user SHALL be able to hide and show the object being placed.

#### Scenario: Add a tree
- **WHEN** the user types "a mature oak tree" and confirms placement
- **THEN** a new version contains the original scene plus the tree at the chosen position and scale

#### Scenario: Cancel placement
- **WHEN** the user cancels before confirming
- **THEN** no new version is created

#### Scenario: Click to place
- **WHEN** the user clicks a point on the ground of the scene during placement
- **THEN** the object's base moves to that point and the position sliders show the new values

#### Scenario: Drag
- **WHEN** the user drags the object in the viewer and releases it
- **THEN** the object follows the pointer at its current height, and "Place in new version" uses the released position

#### Scenario: Hide while placing
- **WHEN** the user turns off "Show object" during placement
- **THEN** the object is hidden in the viewer and the placement values are kept; confirming still adds the object
