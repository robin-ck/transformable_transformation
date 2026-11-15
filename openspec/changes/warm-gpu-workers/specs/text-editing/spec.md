# Delta: text-editing

## ADDED Requirements

### Requirement: Fast response for object and preview jobs
The system SHALL keep GPU workers for Add object and Preview loaded between calls for a limited idle time, and SHALL start them in the background when the user opens a scene or one of these tools. Scene creation is excluded.

#### Scenario: Warm add object
- **WHEN** the user opened a scene at least 1 minute ago and requests "a mature oak tree"
- **THEN** the object is ready to place within 30 seconds

#### Scenario: Idle shutdown
- **WHEN** no Add object or Preview job runs for the idle window (default 5 minutes)
- **THEN** the GPU worker shuts down and no further GPU time is billed

#### Scenario: Warm-up fails
- **WHEN** the background warm-up cannot start
- **THEN** the user sees no error, and the next job starts as it does today
