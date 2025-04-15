from sphinx.util.docutils import SphinxDirective


class InvisibleAltairPlot(SphinxDirective):
    has_content = True

    def run(self):
        import altair as alt
        from docutils import nodes

        code = "\n".join(self.content)
        ns = {}
        exec(code, ns)

        chart = ns.get("chart")
        if not isinstance(chart, alt.TopLevelMixin):
            raise ValueError("Expected a variable named 'chart' with an Altair chart object.")

        chart_id = f"vega-spec-{id(chart)}"
        spec = chart.to_json(indent=None)

        html = f"""
        <div class="altair-chart">
            <script type="application/json" id="{chart_id}">
                {spec}
            </script>
            <script>
                const el = document.getElementById("{chart_id}");
                const spec = JSON.parse(el.textContent);
                vegaEmbed(el.parentElement, spec, {{
                    actions: {{
                        export: true,
                        source: true,
                        editor: true,
                        compiled: false
                    }}
                }});
            </script>
        </div>
        """

        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_directive("invisible-altair-plot", InvisibleAltairPlot)
