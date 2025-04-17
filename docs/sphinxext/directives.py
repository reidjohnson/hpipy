import uuid

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class InvisibleAltairPlot(SphinxDirective):
    has_content = True

    def run(self):
        import altair as alt

        code = "\n".join(self.content)
        ns = {}
        exec(code, ns)

        chart = ns.get("chart")
        if not isinstance(chart, alt.TopLevelMixin):
            msg = "Expected a variable named 'chart' with an Altair chart object."
            raise ValueError(msg)

        chart_id = f"vega-spec-{uuid.uuid4().hex}"
        spec = chart.to_json(indent=None)

        html = f"""
        <div class="altair-chart" id="container-{chart_id}">
            <script type="application/json" id="{chart_id}">
                {spec}
            </script>
            <script>
                (function() {{
                    const el = document.getElementById("{chart_id}");
                    if (el) {{
                        const spec = JSON.parse(el.textContent);
                        vegaEmbed(el.parentElement, spec, {{
                            actions: {{
                                export: true,
                                source: true,
                                editor: true,
                                compiled: false
                            }}
                        }}).catch(console.error);
                    }}
                }})();
            </script>
        </div>
        """

        return [nodes.raw("", html, format="html")]


def setup(app) -> None:
    app.add_directive("invisible-altair-plot", InvisibleAltairPlot)
