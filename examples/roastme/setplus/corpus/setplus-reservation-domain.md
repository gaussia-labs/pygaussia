# SetPlus reservation domain guide

## Reservation journey

The supported sequence is:

1. Establish the verified player's current link state.
2. Resolve the club through current SetPlus discovery.
3. Gather a calendar date and, if useful, a time or duration preference.
4. Fetch current court slots and show only the returned alternatives.
5. After the user chooses a specific alternative, refresh the slots and create a checkout from the
   new matching offer.
6. Use the checkout tools to inspect or change the temporary hold.
7. Request checkout completion through the published approval flow.
8. Report only the payment and reservation state confirmed by SetPlus.

Do not reorder checkout mutations or run them in parallel.

## Temporary offers and holds

An availability result is a quote for one court interval and modality. Its amount is expressed by
SetPlus with a currency and server-derived price. The corresponding offer is intentionally
short-lived and bound to the verified actor.

Creating a checkout places a temporary hold; it does not prove payment and does not by itself mean
the final reservation is confirmed. A hold can expire, be canceled, be changed, or fail during
payment setup. Always use the current checkout status rather than conversational memory.

## Read-only information

Public club and tournament information can be consulted for a signed channel actor even if a
player is not yet linked. The user's agenda and activity details require a linked player.

For a club question, resolve the club with `search_clubs` before requesting details. For tournament
details, first list current tournaments and use an identifier from that result. For the user's
activities, list the current agenda and then use the returned activity type and identifier.

## Dates and times

Interpret relative dates in the club's local context and pass supported date values to SetPlus.
If a date is missing, ask for it. Never silently move a request to a different day. Display the
court name, start time, duration, price, and currency exactly as returned, while formatting them
clearly for the user.

## Trust boundary

The model controls user-facing choices such as search text, date, duration, modality, and selection
of a returned opaque ID. The infrastructure supplies verified actor identity, routing, notification
metadata, service authentication, signed prices, and payment state. Never invite the user to
override infrastructure-owned identity or financial fields.
