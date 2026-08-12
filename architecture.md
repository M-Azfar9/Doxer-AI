```
                    USER
                      │
                      ▼
                ┌───────────┐
                │   Router  │
                └─────┬─────┘
                      │
          ┌───────────┴───────────┐
          │                       │
     Static Concept          Live Research
          │                       │
          │                ┌──────▼──────┐
          │                │ Researcher  │
          │                └──────┬──────┘
          │                       │
          │              ┌────────┴────────┐
          │              │                 │
          │           Tavily            GitHub
          │              │                 │
          │              └────────┬────────┘
          │                       │
          └──────────┬────────────┘
                     ▼
              Evidence Store
                     │
                     ▼
              ┌──────────────┐
              │ Orchestrator │
              └──────┬───────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
 Architecture     API Docs     Installation
 Worker           Worker       Worker
       │             │             │
       └─────────────┼─────────────┘
                     ▼
              Reducer / Reviewer
                     │
                     ▼
              Diagram Generator
                     │
                     ▼
               Final DevDocs
```



