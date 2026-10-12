# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel, Field

MODEL = "gemini-3.8-flash"


class QueryClassification(BaseModel):
    """Structured output schema for classifying user inquiries."""

    category: Literal["shipping", "unrelated"] = Field(
        description=(
            "Classification of the customer's query: 'shipping' if it concerns "
            "shipping rates, tracking, delivery status, or returns; 'unrelated' otherwise."
        )
    )
    explanation: str = Field(
        default="",
        description="Brief reasoning for the chosen classification category.",
    )


# 1. Classification Node (LLM Agent)
classifier_agent = LlmAgent(
    name="classifier_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an intent classification specialist for a customer support shipping company.\n"
        "Your task is to classify incoming customer inquiries into one of two categories:\n"
        "1. 'shipping': Any inquiry related to shipping services, including rates/pricing, "
        "package tracking, delivery schedules/status, shipping options, lost packages, or returns/exchanges.\n"
        "2. 'unrelated': Any query not related to shipping operations (general trivia, unrelated products, etc.).\n\n"
        "Be helpful and inclusive when determining if an inquiry touches on shipping."
    ),
    output_schema=QueryClassification,
)


# 2. Router Node (Function Node)
def route_classification(node_input: dict) -> Event:
    """Routes the workflow based on query classification result.

    Args:
        node_input: Dict from classifier_agent containing the 'category' field.

    Returns:
        Event specifying the routing branch ('shipping' or 'unrelated').
    """
    category = node_input.get("category", "unrelated")
    if category == "shipping":
        return Event(route="shipping")
    return Event(route="unrelated")


# 3. Shipping FAQ Node (LLM Agent)
shipping_faq_agent = LlmAgent(
    name="shipping_faq_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a friendly, professional customer support representative and shipping FAQ specialist "
        "for a shipping company.\n"
        "Your role is to assist customers with all aspects of shipping, including:\n"
        "- Shipping rates and cost estimates\n"
        "- Package tracking and shipment progress\n"
        "- Delivery times, options, and delays\n"
        "- Returns, reverse logistics, and refund processes\n\n"
        "Provide clear, accurate, reassuring, and concise assistance. If tracking or order numbers are "
        "provided, acknowledge them and explain next steps clearly."
    ),
)


# 4. Decline Handler Node (Polite rejection for unrelated queries)
def decline_handler() -> Event:
    """Politely declines inquiries unrelated to shipping operations.

    Returns:
        Event with a polite declination message and user-facing content.
    """
    decline_message = (
        "Thank you for contacting customer support. We are a customer service assistant "
        "for a shipping company and can only assist with shipping-related matters, such as "
        "shipping rates, package tracking, delivery schedules, and returns. "
        "Please let us know if you have any questions regarding your shipments or shipping services!"
    )
    return Event(
        output=decline_message,
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=decline_message)],
        ),
    )


# Graph Workflow Definition
root_agent = Workflow(
    name="customer_support_agent",
    description=(
        "Customer support workflow agent for a shipping company that classifies queries "
        "into shipping or unrelated categories, routing shipping inquiries to a shipping FAQ agent "
        "and politely declining unrelated queries."
    ),
    edges=[
        ("START", classifier_agent),
        (classifier_agent, route_classification),
        (
            route_classification,
            {
                "shipping": shipping_faq_agent,
                "unrelated": decline_handler,
            },
        ),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
