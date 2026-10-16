# Customizing LiteratureAgent for a new scientific domain

Use the **General science** profile for a non-perovskite project unless a
maintainer has added and tested a dedicated profile. Before processing a large
collection, decide what the future database must represent and validate the
result on a few familiar papers.

## Four different things you may need to change

These files and concepts are related, but they are not interchangeable.

| Concern | What it controls | Starting point |
|---|---|---|
| Domain ontology | Preferred scientific terms, synonyms, and knowledge-graph normalization vocabulary | `canonical_terms` in a copy of `config/domain_packs/custom_domain_template.json` |
| Structured-output contract | Which fields are extracted, their names, shapes, and evidence requirements | `structured_output` in the same domain-pack JSON |
| Database design | Row grain, identifiers, types, units, allowed values, and tables used after export | Plan this before changing the extraction contract; start from `csv/domain_records.csv` |
| Knowledge-graph mapping | Which structured fields become nodes and relationships | `add_domain_record` in `scripts/build_literature_knowledge_graph.py` |

Changing `canonical_terms` or broad concept lists does **not** add a CSV column
or change the JSON output shape; only `structured_output.fields` does that.
Custom fields receive a safe generic graph mapping, but richer typed graph
semantics remain a separate design decision. Keep all four layers aligned.

The file `config/literatureagent_kg_schema_v1.json` documents the governed
integrated graph contract. Editing that JSON alone does not implement a new
runtime mapping in the standalone graph builder.

## 1. Define the database before changing extraction

Write down the following decisions for the new material system or scientific
field:

- the row grain, such as one paper, one sample, one formulation, one experiment,
  or one measurement condition;
- stable identifiers for papers and for the row-level scientific object;
- each field name, description, scalar/list/object shape, and required type;
- canonical units and whether source-reported units are also retained;
- controlled vocabularies and synonym rules;
- the difference between missing, not applicable, not measured, and not
  reported;
- which fields require a verbatim source excerpt;
- how repeated measurements and multiple samples in one paper are represented;
- uniqueness, duplicate-review, and human-adjudication rules.

Do not turn `data/Perovskite_database_content_all_data.csv` into the new-domain
database. That bundled baseline and the model under `models/` are compatibility
resources for the **halide-perovskite photovoltaic** workflow only. General
science results are exported independently.

## 2. Create the domain pack and ontology

Copy `config/domain_packs/custom_domain_template.json` to a clearly named JSON
file, such as `config/domain_packs/battery_materials.json`. Do not overwrite the
template. Update its `id`, `version`, `description`, `customization_notes`, and
concept lists, and keep it valid UTF-8 JSON.

Include the preferred names, synonyms, abbreviations, alternate spellings,
methods, conditions, properties, and units found in the target literature.
Terms that should canonicalize to one knowledge-graph entity belong in the
top-level `canonical_terms` list:

```json
"canonical_terms": [
  {
    "canonical": "lithium iron phosphate",
    "aliases": ["LFP", "LiFePO4"],
    "category": "cathode_material"
  }
]
```

`synonyms` may be used instead of `aliases`. Broad vocabulary lists and any
`structured_output` metadata are deliberately not collapsed to their parent
category names by the graph normalizer.

Use source control and increment the version whenever vocabulary or output
fields change. LiteratureAgent writes `_domain_pack_id`,
`_domain_pack_version`, and `_domain_pack_file` provenance into each domain
record and carries the identifiers onto Paper and StudiedSystem graph nodes.
In the application choose **General science**, open **Advanced options**, and
select the copied file under **Ontology / domain-pack JSON**. A copied file does
not appear as a separate profile name; adding a dedicated profile label still
requires a maintainer to register and test it.

## 3. Review or extend the structured output

The built-in General science contract already captures:

- study type and studied system or subject;
- materials or scientific entities;
- intervention or variable and comparison;
- methods and conditions;
- measurements with value, unit, context, and evidence;
- primary findings with evidence and confidence;
- mechanisms and limitations.

Its compact table is `csv/domain_records.csv`; the compatibility table is
`csv/all_records.csv`. Raw structured responses and paper summaries remain in
the run folder for provenance and review.

If the built-in fields are sufficient, remove or leave empty the template's
example custom fields and adapt only the vocabulary. If the database needs more
fields, edit `structured_output` in the copied domain pack:

- `domain_instruction` tells the extractor what scientific distinctions matter;
- every key under `fields` must use lower-case `snake_case`, be at most 64
  characters, and becomes a `Domain_<field_name>` CSV column;
- the example value defines the expected shape: `null` for a scalar, `[]` for a
  list, `{}` for an object, or a list containing an example object;
- scalar and simple-list fields need a verbatim excerpt in `field_evidence`;
- each retained object in a measurement or outcome list needs its own
  `evidence` member;
- do not configure `field_evidence` itself; LiteratureAgent manages it;
- add tests with present, missing, repeated, malformed, and unsupported values
  when the contract will be maintained as code.

Configured fields extend the built-in General science contract and are exported
automatically; editing the large controller is not required for ordinary custom
fields. Code changes are still appropriate for validation rules or semantics
that cannot be represented by the JSON shapes.

Field-level evidence is part of the data contract, not optional decoration. A
populated factual value without a matching excerpt from the supplied paper text
is rejected.

## 4. Keep the knowledge graph domain-safe

The standard cross-domain graph uses Paper, StudiedSystem, DomainEntity, Method,
ExperimentalCondition, Measurement, ScientificClaim, and Evidence nodes. It
must not create photovoltaic Device, PCE, or stability-model nodes for an
unrelated domain.

The existing General science fields map automatically. Any other populated
`Domain_*` column is retained generically as a `DomainAttribute` node connected
to the studied system by `HAS_DOMAIN_ATTRIBUTE`. A matching
`_evidence_<exact Domain column>` is linked to that node. This fallback keeps a
new domain graph-complete without inventing photovoltaic semantics.

For every new structured field, still decide whether the generic representation
is sufficient or whether it should be:

- a property on an existing node;
- a new entity or measurement node;
- a relationship between existing nodes; or
- tabular provenance that should not become graph topology.

If richer typed semantics are required, update `add_domain_record` in
`scripts/build_literature_knowledge_graph.py`; preserve the source field and
record identifier on relationships and connect the matching evidence node. Add
a test that builds the graph from a small non-perovskite CSV and asserts the
expected canonical names, generic or typed relationships, evidence links, and
absence of photovoltaic-only nodes.

The graph builder writes:

- `knowledge_graph/neo4j_import/nodes.csv`;
- `knowledge_graph/neo4j_import/relationships.csv`;
- reports, plots, derived views, and optional portable JSONL/GraphML exports.

These Neo4j CSV files can seed a graph database; `csv/domain_records.csv` can
seed a relational or analytical database. Neither output replaces database
governance, validation, or migrations.

## Required pilot before a full campaign

Run **one to three papers whose contents you already know**, including at least
one paper with multiple samples or measurements if that is expected in the
domain.

Before scaling up, verify all of the following:

- row grain and identifiers match the planned database;
- expected fields, list/object shapes, types, units, and missing values are
  correct in `csv/domain_records.csv` and the raw JSON;
- every retained factual value has a source excerpt that actually supports it;
- background statements are not recorded as the paper's own results;
- multiple samples and conditions are not incorrectly merged;
- summaries use the new domain's terminology without photovoltaic assumptions;
- the knowledge graph contains the intended domain nodes and evidence links;
- the graph contains no placeholder Device/PCE/stability nodes;
- Neo4j import CSVs open cleanly and have no dangling relationships;
- a domain expert accepts the pilot records.

Only then freeze the ontology and output-contract versions and run the larger
collection. Keep the pilot artifacts as a regression fixture for future
changes.
