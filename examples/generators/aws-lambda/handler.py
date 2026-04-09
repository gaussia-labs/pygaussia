"""AWS Lambda handler for Gaussia generators module."""

import json

from run import run


def lambda_handler(event, context):
    """Lambda handler for API Gateway HTTP API (payload format v2.0).

    Expects JSON in event["body"] (string) and returns JSON response.

    Args:
        event: Lambda event containing the request body
        context: Lambda context object

    Returns:
        dict: Response with statusCode, headers, and body
    """
    body = event.get("body") or "{}"

    # API Gateway sends body as a string
    if isinstance(body, str):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return _response(400, {"error": "Invalid JSON body"})
    else:
        payload = body if body else {}

    try:
        result = run(payload)
    except Exception as e:
        return _response(500, {"error": str(e)})

    return _response(200, result)


def _response(status_code: int, data: dict) -> dict:
    """Build Lambda response object."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(data, default=str),
    }
