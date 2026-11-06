# Tests and hardware evidence

- Never write unit tests after you write code.
- Highly prefer E2E tests as the sole testing mechanism. Use them to verify complex features work. At the end of E2E tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, first write down all the ways it could fail, then write the code.
- Do not remove existing tests without establishing which real failures are covered by the retained E2E checks.
- Keep native tuning bundles, recordings, credentials and machine identifiers out of public test artifacts. Preserve kernel speaker locks and exact gain readback.
