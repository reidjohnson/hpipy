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
                    const container = document.getElementById("container-{chart_id}");
                    const specScript = document.getElementById("{chart_id}");
                    if (!container || !specScript) {{
                        console.error("[AltairPlot] Missing container or spec script!");
                        return;
                    }}

                    let lastTheme = null;
                    let chartDiv = null;

                    function getTheme() {{
                        const dt = document.documentElement.getAttribute("data-theme");
                        return dt === "dark" ? "dark" : "light";
                    }}

                    function renderChart() {{
                        const theme = getTheme();

                        // No need to re-render if theme did not change.
                        if (theme === lastTheme && chartDiv) return;
                        lastTheme = theme;

                        const spec = JSON.parse(specScript.textContent);

                        // Clear previous rendering.
                        container.innerHTML = "";

                        // Create a new div inside container.
                        chartDiv = document.createElement("div");
                        container.appendChild(chartDiv);

                        vegaEmbed(chartDiv, spec, {{
                            theme: theme,
                            actions: {{
                                export: true,
                                source: true,
                                editor: true,
                                compiled: false
                            }},
                            defaultStyle: true
                        }}).catch(console.error);
                    }}

                    document.addEventListener("DOMContentLoaded", renderChart);

                    const observer = new MutationObserver(mutations => {{
                        if (mutations.some(m => m.attributeName === "data-theme")) {{
                            renderChart();
                        }}
                    }});
                    observer.observe(document.documentElement, {{
                        attributes: true,
                        attributeFilter: ["data-theme"]
                    }});
                }})();
            </script>
        </div>
        """

        return [nodes.raw("", html, format="html")]


def setup(app) -> None:
    app.add_directive("invisible-altair-plot", InvisibleAltairPlot)
