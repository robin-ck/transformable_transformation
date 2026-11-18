# Delta: space-layout

## MODIFIED Requirements

### Requirement: Tools next to the viewer
The Scene tab SHALL show the viewer and a tool panel side by side on desktop widths, with the sub-tabs History, Objects, Edit scene, Preview and Futures that are switched on in the tool visibility options, so the viewer stays visible while a tool is used. On narrow screens the panel MAY stack below the viewer.

#### Scenario: Switch tool
- **WHEN** the user switches from Objects to Preview
- **THEN** the viewer keeps its scene, version and camera

#### Scenario: Placement in progress
- **WHEN** the user switches sub-tab while an object is being placed
- **THEN** the placement is kept and resumes when they return to Objects

#### Scenario: Region box only in Edit scene
- **WHEN** the user has drawn a region box in Edit scene and switches to another sub-tab
- **THEN** the box is removed, the viewer orbits again, and "Limit to a region" is switched off

## ADDED Requirements

### Requirement: Tool visibility options
The Space's Settings dialog (page footer) SHALL offer a "Scene tools" section with one switch per tool sub-tab. A switched-off tool's sub-tab SHALL be hidden; when all are switched off, the tool panel SHALL be hidden and the viewer SHALL use the full width. The choice SHALL be remembered per browser, and all tools SHALL be shown when no choice is stored or the section cannot be added.

#### Scenario: Hide a tool
- **WHEN** the user switches off Futures in Settings
- **THEN** the Futures sub-tab disappears and the other tools are unchanged

#### Scenario: Hide the open tool
- **WHEN** the user switches off the sub-tab that is currently open
- **THEN** the first visible sub-tab opens, and the viewer keeps its scene, version and camera

#### Scenario: Hide all tools
- **WHEN** the user switches off every tool
- **THEN** the tool panel is hidden and the viewer spans the full width without losing its camera

#### Scenario: Show a tool again
- **WHEN** the user switches a hidden tool back on
- **THEN** its sub-tab reappears with the state it had before (for example a placement in progress)

#### Scenario: Return visit
- **WHEN** the user reloads the Space in the same browser
- **THEN** the same tools are hidden as before

#### Scenario: First visit or Settings not available
- **WHEN** no choice is stored, or the Settings dialog cannot be extended
- **THEN** all tools are shown
