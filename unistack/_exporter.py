from datetime import datetime, timezone

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode, format_span_id, format_trace_id
from pymongo import MongoClient


class MongoDBSpanExporter(SpanExporter):
    def __init__(self, mongo_uri: str, db_name: str = "unistack"):
        self._client = MongoClient(mongo_uri)
        self._db = self._client[db_name]

    def export(self, spans):
        try:
            for span in spans:
                ctx = span.context
                trace_id = format_trace_id(ctx.trace_id)
                span_id = format_span_id(ctx.span_id)
                parent_span_id = format_span_id(span.parent.span_id) if span.parent else None

                start_time = datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc)
                end_time = datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc)
                duration_ms = (span.end_time - span.start_time) / 1e6

                attrs = dict(span.attributes or {})
                activity_id = attrs.get("unistack.activity_id", trace_id[:16])
                workflow = attrs.get("unistack.workflow", span.name)

                unistack_status = attrs.get("unistack.status")
                if unistack_status:
                    status = unistack_status
                elif span.status.status_code == StatusCode.ERROR:
                    status = "failed"
                else:
                    status = "completed"

                self._db.spans.insert_one({
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "parent_span_id": parent_span_id,
                    "name": span.name,
                    "activity_id": activity_id,
                    "workflow": workflow,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_ms": duration_ms,
                    "status": status,
                    "attributes": attrs,
                })
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self):
        self._client.close()
