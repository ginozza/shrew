# Introduction to Shrew

Shrew files usually have the extension `.sw`. A Shrew program consists of a series of **directives** that define metadata, configuration, types, and computation graphs.

## File Structure

A typical `.sw` file structure looks like this:

```sw
// Metadata about the model
@model { ... }

// Training or Inference configuration
@config { ... }

// Type definitions (optional aliases)
@types { ... }

// Computations graphs (functions)
@graph MyGraph(...) { ... }
```

## Comments

Shrew supports C-style comments:
- Single line: `// comment`
- Multi-line: `/* comment */`
