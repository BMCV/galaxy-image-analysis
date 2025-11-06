PLEASE DESCRIBE YOUR PR HERE

---

### Check-list for the contributor

Please make sure you have read the [CONTRIBUTING.md](https://github.com/BMCV/galaxy-image-analysis/blob/master/CONTRIBUTING.md) document (last updated: 2024/04/23).

**Please fill out if applicable:**

* [ ] License permits unrestricted use (educational + commercial).

**If this PR adds or updates a tool or tool collection:**

* [ ] This PR adds a new tool or tool collection.
* [ ] This PR updates an existing tool or tool collection.
* [ ] Tools added/updated by this PR comply with the Guidelines below (or explain why they do not).

### Guidelines for the contributor

<details>
<summary>
    This section is cited from the <a href="https://doi.org/10.37044/osf.io/w8dsz">Naming and Annotation Conventions for Tools in the Image Community in Galaxy</a>.
</summary>

<h4>Naming</h4>

Generally, the name of Galaxy tools in our community should be expressive and concise, while stating the purpose of the tool as precisely as possible. Consistency of the namings of Galaxy tools is important to ensure they can be found easily. To maintain consistency, we consider phrasing names as imperatives a good practice, such as "Analyze particles" or "Perform segmentation using watershed transformation". An acknowledged exception from this rule is the names of tool wrappers of major tool suites, where the name of a tool wrapper should be chosen identically to the module or function of the tool which is wrapped (e.g., "MaskImage" in CellProfiler).

<h4>Tool description</h4>

If a Galaxy tool is a thin tool wrapper (e.g, part of a major tool suite), then the name of the wrapped tool (and only the name of the wrapped tool, subsequent to the term "with" as in "with Bioformats") should be used as the description of the tool (further examples include "with CellProfiler", "with ImageJ2", "with ImageMagick", "with SpyBOAT", "with SuperDSM"). This ensures that the tool is found by typing the name of the wrapped tool into the "Search" field on the Galaxy interface. The tool description should be empty if a tool is either not part of a major tool suite, or the main functionality of the tool is implemented in the wrapper.

<h4>Annotations</h4>

We point out that there is a global list of precedential annotations with <a href="https://bio.tools">Bio.tools</a> identifiers (Ison et al., 2019) in Galaxy (see <a href="https://github.com/galaxyproject/galaxy/blob/dev/lib/galaxy/tool_util/ontologies/biotools_mappings.tsv">mappings</a>), which may outweigh the annotations made in the XML specification of a Galaxy tool (and thus the annotations of a tool reported within the web interface of Galaxy might be divergent). However, since the precedential annotations are subject to possible changes and to avoid redundant work, we do not aim to reflect those in our specifications (those which we make in the XML specifications of Galaxy tools).

</details>
