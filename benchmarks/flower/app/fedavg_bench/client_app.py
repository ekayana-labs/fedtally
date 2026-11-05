"""ClientApp: one call to the shared training step per round."""

from common import local_update
from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    params = {name: array.numpy() for name, array in msg.content["arrays"].items()}
    update = local_update(params, int(context.node_config["site"]))
    content = RecordDict(
        {
            "arrays": ArrayRecord({name: Array(value) for name, value in update.items()}),
            "metrics": MetricRecord({"num-examples": 1}),
        }
    )
    return Message(content=content, reply_to=msg)
