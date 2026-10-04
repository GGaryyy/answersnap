"""HTML rendering: one self-contained file, no JavaScript, autoescaped.

Answer text comes from language models and from web pages, so it is untrusted
input; every value goes through Jinja's autoescape rather than hand-built
strings.
"""

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from answersnap.report import format as fmt


def _environment():
    env = Environment(loader=PackageLoader("answersnap", "report/templates"),
                      autoescape=select_autoescape(["html", "j2"], default=True),
                      undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    env.globals.update(metric_text=fmt.metric_text, has_value=fmt.has_value, pct=fmt.pct,
                       interval=fmt.interval, faith_summary_text=fmt.faith_summary_text,
                       faith_row_label=fmt.faith_row_label, yes_blank=fmt.yes_blank,
                       engine_status_text=fmt.engine_status_text,
                       missing_metric_text=fmt.missing_metric_text,
                       low_sample=fmt.LOW_SAMPLE_NOTE)
    return env


def render_html(report):
    return _environment().get_template("report.html.j2").render(report=report)
