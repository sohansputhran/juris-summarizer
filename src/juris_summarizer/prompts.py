MAP_PROMPT_TEMPLATE = """You are summarizing one section of a longer academic paper. This is section {chunk_num} of {total_chunks}.

Write a concise summary of ONLY the content in this section. Do not add information not present in the text. Do not refer to "this section" or "this chunk" — write as if summarizing standalone content.

Section text:
{chunk_text}

Summary:"""


REDUCE_PROMPT_TEMPLATE = """You are writing the final summary of an academic paper, based on summaries of its individual sections below.

Combine the section summaries into a single, coherent, densely-written abstract of the paper. Remove redundancy across sections. Do not simply concatenate the section summaries — synthesize them into a unified narrative that reads as if written by the paper's author. Do not add information not present in the section summaries.

Here is an example of the input format and the expected output style:

Example section summaries:
{example_chunk_summaries}

Example final summary:
{example_reference_summary}

Now do the same for the following paper.

Section summaries:
{combined_summaries}

Final summary:"""


def format_chunk_summaries(summaries: list[str]) -> str:
    """Shared formatting so the example and the real task use an identical input shape."""
    return "\n\n".join(f"Section {i+1}: {s}" for i, s in enumerate(summaries))
