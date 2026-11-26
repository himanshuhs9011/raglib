# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** (GitHub private vulnerability reporting). Do not open a public issue, pull request or discussion for a vulnerability.

Include what you can:

- the raglib version and the configuration involved (`rag_type`, vector store, providers, guards);
- what an attacker can do, and the steps or a minimal script to reproduce it;
- whether it needs a malicious document, question, URL, file or saved index.

You get an acknowledgement within a week. Fixed issues are published as a GitHub security advisory and listed under **Security** in the [CHANGELOG](CHANGELOG.md), with credit if you want it.

## Supported versions

| Version | Fixes |
| :--- | :--- |
| Latest minor release (2.3.x once released, 2.2.x until then) | All bug and security fixes |
| The minor release before it | Security fixes only, until the next minor release |
| Older | None. Please upgrade. |

## Scope

In scope: anything in the `raglib-py` package that lets input it processes (documents, questions, URLs, files, saved indexes, model output) leak data across users or tenants, bypass `filters=` or `access_control`, run code, reach internal network addresses, or exhaust resources despite the documented limits.

Out of scope: vulnerabilities in the LLM, embedding or search providers themselves, in vector-store servers you run (for example a Chroma or Qdrant server), and attacks that need control of the machine running raglib. The README section [Security & Privacy](README.md#security--privacy) lists what raglib does and what you should do yourself.
