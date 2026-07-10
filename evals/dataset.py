"""Eval cases: fixed inputs with expectations."""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    description: str
    user_message: str
    auth: str | None = None  # "user" = signed in as the eval user; None = signed out
    history: list[tuple[str, str]] = field(
        default_factory=list
    )  # (role, text): "user"|"ai"

    # Expectations - used by the scorers.
    expect_tools: list[str] = field(default_factory=list)
    expect_refusal: bool = False  # should decline / route to login, not act
    must_not_error: bool = True  # no tool should return an error
    reference: str | None = None  # ideal answer, for the judge / exact-match
    rubric: str | None = None  # or a rubric, for the judge
    # Judge that the reply only uses facts from the tool results. Use for answers
    # synthesized from retrieved content (FAQ, inventory); not for structured
    # outputs (identity, orders), which are checked deterministically instead.
    check_grounding: bool = False


CASES: list[EvalCase] = [
    EvalCase(
        id="faq_international",
        description="FAQ answered from the knowledge base",
        user_message="Do you ship flowers internationally?",
        expect_tools=["search_business_faqs"],
        check_grounding=True,
    ),
    EvalCase(
        id="faq_damaged",
        description="FAQ: what to do about a damaged delivery",
        user_message="What should I do if my flowers arrive damaged?",
        expect_tools=["search_business_faqs"],
        check_grounding=True,
    ),
    EvalCase(
        id="inventory_budget",
        description="Find products under a budget",
        user_message="What do you have for under $50?",
        expect_tools=["search_product_inventory"],
        check_grounding=True,
    ),
    EvalCase(
        id="inventory_occasion",
        description="Find a product for an occasion within a budget",
        user_message="I need a bright, cheerful bouquet for a birthday, around $70.",
        expect_tools=["search_product_inventory"],
        check_grounding=True,
    ),
    EvalCase(
        id="who_am_i",
        description="Identity lookup when signed in",
        user_message="What's the email on my account?",
        auth="user",
        expect_tools=["get_current_user"],
        reference="eval-user@example.com",  # exact-match target (stage 2)
    ),
    EvalCase(
        id="get_orders",
        description="Order history when signed in",
        user_message="Can you show me my past orders?",
        auth="user",
        expect_tools=["get_orders"],
    ),
    EvalCase(
        id="order_happy",
        description="Place an order after the customer confirms",
        # The agent confirms before ordering, so the confirmation already happened
        # in history; this turn is the customer approving it.
        history=[
            (
                "user",
                "I'd like to order 1 Crystal Clear Glass Vase, "
                "delivered to Shankhamul, Kathmandu.",
            ),
            (
                "ai",
                "To confirm: 1 Crystal Clear Glass Vase to Shankhamul, Kathmandu. "
                "Shall I place the order?",
            ),
        ],
        user_message="Yes, please place the order.",
        auth="user",
        expect_tools=["place_order"],
    ),
    EvalCase(
        id="order_out_of_stock",
        description="Asked for more than is in stock -> explains, does not order",
        user_message="Order 100 of the Vibrant Sunflower Surprise to Kathmandu.",
        auth="user",
        expect_refusal=True,  # should not place an order it can't fulfill
        rubric="Tells the customer there isn't enough stock and does not confirm an order.",
    ),
    EvalCase(
        id="order_signed_out",
        description="Ordering while signed out -> routed to login, nothing ordered",
        user_message="Order 1 Crystal Clear Glass Vase to Shankhamul, Kathmandu.",
        auth=None,
        expect_refusal=True,  # place_order must NOT appear in the trajectory
    ),
    EvalCase(
        id="off_topic",
        description="Off-topic request politely declined",
        user_message="Can you write me a poem about quantum physics?",
        expect_refusal=True,
    ),
]
