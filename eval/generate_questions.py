#!/usr/bin/env python3
"""
generate_questions.py

Builds eval/questions.csv -- the fixed 100-question CRM benchmark used by
evaluate.py. You normally don't need to re-run this; it's here so the set
can be regenerated or extended deterministically instead of hand-edited
into an inconsistent state. Editing questions.csv directly is also fine
(it's a plain CSV: id,category,question,rubric).

Run:
    python generate_questions.py > questions.csv
    # or, to write the file directly:
    python generate_questions.py --out questions.csv
"""

import argparse
import csv
import io
import sys

# Sample entities used to make questions concrete. These are plausible
# Frappe CRM record names/values, not real data -- the agent is expected
# to look records up (or fail gracefully) rather than the eval assuming
# any of them exist in a given CRM instance. See README for how rubrics
# are graded without ground-truth DB access.
LEADS = ["Amit Shah", "Priya Nair", "Elena Novak", "Kevin Ortiz", "Lena Wu"]
DEALS = ["Northwind Logistics", "Blue Harbor Retail", "Atlas Manufacturing", "Quantum Foods", "Vertex Health"]
ORGS = ["Northwind Logistics", "Cedar & Co", "Orbital Systems", "Marigold Textiles", "Pinehill Labs"]
CONTACTS = ["Rahul Mehta", "Sara Kim", "Daniel Obi", "Fatima Al-Sayed", "Tom Bennett"]
TASKS = ["Follow up on proposal", "Send updated quote", "Schedule demo call", "Confirm PO details", "Renewal check-in"]

rows = []
_id = 0


def add(category, question, rubric):
    global _id
    _id += 1
    rows.append({"id": _id, "category": category, "question": question, "rubric": rubric})


# ---------------------------------------------------------------------------
# 1. crm_search -- retrieval / filtering (20)
# ---------------------------------------------------------------------------
add("crm_search", "Show me all leads with status 'New'.",
    "Should call crm_search on doctype CRM Lead filtered by status=New (one of Frappe CRM's default lead statuses: New, Contacted, Nurture, Qualified, Unqualified, Junk), then summarize the returned records only. Must not invent lead names not present in the tool output; if the tool returns zero rows, must say so plainly rather than guessing.")
add("crm_search", "Which deals are currently in the 'Negotiation' status?",
    "Should call crm_search on CRM Deal filtered on the status field (Frappe CRM deals use a status field, not a separate 'stage' field -- 'Negotiation' is one of the out-of-box deal statuses), report only records actually returned, and state the count.")
add("crm_search", "Find every organization in the 'Manufacturing' industry.",
    "Should call crm_search on CRM Organization filtered by industry, and list only records the tool actually returned.")
add("crm_search", "List contacts whose email domain is 'northwind.com'.",
    "Should use crm_search on Contact with an appropriate filter/search_text; if the tool has no direct domain filter it should say what filter it used or that it searched by text match, without fabricating matches.")
add("crm_search", "Search for any deal related to 'Blue Harbor Retail'.",
    "Should call crm_search on CRM Deal with search_text or a name/organization filter for Blue Harbor Retail, and report actual results (including 'none found' if applicable).")
add("crm_search", "What tasks are overdue right now?",
    "Should call crm_search on CRM Task with a date/status filter for overdue items, and only report what the tool returns; should not assume today's date without checking or state one confidently if the tool doesn't support date comparison directly.")
add("crm_search", "Give me the 5 most recently created leads.",
    "Should call crm_search on CRM Lead with an appropriate sort/order (e.g. creation desc) and limit, then list exactly what's returned.")
add("crm_search", "Are there any leads with no owner assigned?",
    "Should call crm_search on CRM Lead filtered for an empty/null owner field, and report actual results without guessing.")
add("crm_search", "Find deals worth more than $50,000.",
    "Should call crm_search on CRM Deal with a numeric filter on the value/amount field, and report only what's returned; should ask which currency/field if genuinely ambiguous rather than guessing silently.")
add("crm_search", "Which leads came from the 'Website' source?",
    "Should call crm_search on CRM Lead filtered by source=Website and report actual results.")
add("crm_search", "Search contacts named 'Sara'.",
    f"Should call crm_search on Contact with search_text or a name filter for 'Sara', and report only what the tool returns (may or may not include {CONTACTS[1]}).")
add("crm_search", "List every call log from this week.",
    "Should call crm_search on CRM Call Log with a date-range filter, and report only actual results; should not fabricate call records.")
add("crm_search", "Which organizations don't have any linked deals?",
    "This requires combining crm_search (CRM Organization) with crm_linked_records per org, or clearly stating the tools don't support a direct 'no linked deals' filter and explaining what it checked instead. Should not fabricate an answer.")
add("crm_search", "Show me all notes that mention 'renewal'.",
    "Should call crm_search on CRM Note with search_text='renewal' (or similar) and report only actual matches.")
add("crm_search", "Find leads that are marked as 'Qualified' and came from a referral.",
    "Should call crm_search on CRM Lead with combined filters on status and source; report only actual results.")
add("crm_search", "What's the total count of deals currently open (not won or lost)?",
    "Should call crm_search on CRM Deal filtered to exclude Won/Lost statuses, and report the count returned by the tool rather than an estimate.")
add("crm_search", "List contacts that are not linked to any organization.",
    "Should call crm_search on Contact with a filter for an empty organization field, or explain if that isn't directly filterable, without fabricating a list.")
add("crm_search", "Search for tasks assigned to me that are due today.",
    "Should call crm_search on CRM Task filtered by assignee and due date; if the agent doesn't know who 'me' is, it should ask for clarification (a user/email) rather than guessing an owner.")
add("crm_search", "Which deals have had no activity in the last 30 days?",
    "Requires crm_search on CRM Deal plus crm_activities per deal, or a clear explanation of the approach taken; should not invent a stale-deal list without checking.")
add("crm_search", "Show every lead currently marked as 'Junk'.",
    "Should call crm_search on CRM Lead filtered by status=Junk and report only actual results.")

# ---------------------------------------------------------------------------
# 2. crm_get -- single-record read (10)
# ---------------------------------------------------------------------------
for i, name in enumerate(LEADS[:4]):
    add("crm_get", f"Show me the full details of the lead '{name}'.",
        f"Should call crm_get on CRM Lead for '{name}' (likely after resolving the record via search if the exact doc name isn't known) and present the returned fields; must not invent field values, and should say clearly if the record isn't found.")
for i, name in enumerate(DEALS[:3]):
    add("crm_get", f"What's the current status and value of the '{name}' deal?",
        f"Should locate and call crm_get on CRM Deal for '{name}', then report only the status/value fields actually returned (Frappe CRM deals use a status field, not 'stage'); must say if the record can't be found rather than guessing a status.")
for i, name in enumerate(ORGS[:3]):
    add("crm_get", f"Give me a summary of the organization '{name}'.",
        f"Should call crm_get on CRM Organization for '{name}' and summarize actual returned fields; should not fabricate details (industry, size, etc.) that weren't in the tool output.")

# ---------------------------------------------------------------------------
# 3. crm_create (15)
# ---------------------------------------------------------------------------
add("crm_create", "Create a new lead for John Carter, email john.carter@example.com, from a trade show.",
    "Should call crm_create on CRM Lead with lead_name, email, and source populated from the given details, and confirm creation with the resulting record name/ID. Should not silently drop the source field.")
add("crm_create", "Add a new organization called 'Silverline Freight' in the Logistics industry.",
    "Should call crm_create on CRM Organization with organization_name='Silverline Freight' and industry='Logistics', confirming success and the created record's identifier.")
add("crm_create", "Create a task to follow up with Northwind Logistics next Friday.",
    "Should call crm_create on CRM Task with a title/description referencing the follow-up and a due date resolved to next Friday's actual date; if it can't confirm today's date it should say what date it used, not silently guess wrong.")
add("crm_create", "Add a new contact: Maria Lopez, maria.lopez@cedarco.com, phone +1-555-0199.",
    "Should call crm_create on Contact with the given name, email and phone populated, and confirm the created record.")
add("crm_create", "Log a call: I spoke with Rahul Mehta today about pricing, 15 minutes, went well.",
    "Should call crm_create on CRM Call Log (optionally linked via reference_doctype/reference_docname to the contact) capturing duration and summary, and confirm creation.")
add("crm_create", "Create a deal for Atlas Manufacturing worth $120,000 with status 'Proposal/Quotation'.",
    "Should call crm_create on CRM Deal with organization, value, and status fields populated from the request (Frappe CRM deals use a status field, not 'stage'), and confirm creation with the resulting record identifier.")
add("crm_create", "Add a note to the Northwind Logistics deal: 'Client wants a revised SOW by next week.'",
    "Should call crm_create on CRM Note with the given content and reference_doctype/reference_docname pointing at the Northwind Logistics deal (resolved via search if needed), confirming success.")
add("crm_create", "Create a lead for a company called 'Reef Analytics' with no contact person yet.",
    "Should call crm_create on CRM Lead with the organization/company name populated and should not fabricate a fake contact name just because one wasn't given; may ask for a contact name if the field is required.")
add("crm_create", "Add a task for me to prepare the quarterly business review deck by the 20th.",
    "Should call crm_create on CRM Task with a title and a due date resolved to the 20th of the relevant month; should ask which month if genuinely ambiguous rather than guessing silently.")
add("crm_create", "Create a new user account for a teammate, jordan.lee@example.com, first name Jordan.",
    "Should call crm_create_user (or crm_create on the appropriate doctype) with the given email and first name, and confirm the result, including any welcome-email behavior if relevant.")
add("crm_create", "Log that Sara Kim called in asking about renewal terms, mark it as high priority.",
    "Should call crm_create on CRM Call Log capturing the summary and, if the doctype supports it, priority; should not drop the priority detail silently.")
add("crm_create", "Add Vertex Health as a new deal, no value yet, status should be 'Qualification'.",
    "Should call crm_create on CRM Deal with organization='Vertex Health' and status='Qualification' (Frappe CRM deals use a status field, not 'stage'), leaving the value field unset rather than inventing a number.")
add("crm_create", "Create a contact for Daniel Obi and link him to the Orbital Systems organization.",
    "Should call crm_create on Contact with Daniel Obi's details and a link to the Orbital Systems organization (resolved via search if the exact record name is needed); should confirm the link was set, not just the contact.")
add("crm_create", "Add a follow-up task on the Blue Harbor Retail deal for tomorrow.",
    "Should call crm_create on CRM Task with reference_doctype/reference_docname pointing at the Blue Harbor Retail deal and a due date resolved to tomorrow's actual date.")
add("crm_create", "Create a lead but don't fill in a status -- I'll set that later.",
    "Should call crm_create on CRM Lead, leaving status at whatever the system default is rather than fabricating one, and should ask for at minimum a name/identifying detail if none was given.")

# ---------------------------------------------------------------------------
# 4. crm_update (15)
# ---------------------------------------------------------------------------
add("crm_update", "Move the Northwind Logistics deal to the 'Won' status.",
    "Should locate the Northwind Logistics deal (search/get if needed) and call crm_update on CRM Deal with status='Won' (Frappe CRM deals use a status field, not 'stage'), confirming the change actually applied.")
add("crm_update", "Change Amit Shah's lead status to 'Qualified'.",
    "Should resolve Amit Shah's lead record and call crm_update on CRM Lead with status='Qualified', confirming success.")
add("crm_update", "Update the value of the Atlas Manufacturing deal to $95,000.",
    "Should resolve the deal and call crm_update on CRM Deal with the value/amount field set to 95000, confirming the change.")
add("crm_update", "Mark the 'Send updated quote' task as completed.",
    "Should resolve the task by name/search and call crm_update on CRM Task with a completed status field, confirming success.")
add("crm_update", "Reassign the Quantum Foods deal to a different owner, jordan.lee@example.com.",
    "Should resolve the deal and call crm_update on CRM Deal setting the owner/deal_owner field to jordan.lee@example.com, confirming success.")
add("crm_update", "Fix the phone number on Fatima Al-Sayed's contact record to +971-50-1234567.",
    "Should resolve the contact and call crm_update on Contact with the corrected phone number, confirming the change.")
add("crm_update", "Update Cedar & Co's industry to 'Retail'.",
    "Should resolve the organization and call crm_update on CRM Organization with industry='Retail', confirming success.")
add("crm_update", "Push back the due date on the 'Confirm PO details' task by one week.",
    "Should resolve the task, determine its current due date (via crm_get if needed), and call crm_update with a due date one week later; should not guess a due date without checking the existing one first.")
add("crm_update", "Change Priya Nair's lead source to 'Referral'.",
    "Should resolve the lead and call crm_update on CRM Lead with source='Referral', confirming success.")
add("crm_update", "Set the Marigold Textiles organization's website to www.marigoldtextiles.com.",
    "Should resolve the organization and call crm_update on CRM Organization with the website field set, confirming success.")
add("crm_update", "Update the Vertex Health deal's probability to 80%.",
    "Should resolve the deal and call crm_update on CRM Deal with the probability field set appropriately, confirming success.")
add("crm_update", "Mark Kevin Ortiz's lead as lost with reason 'Budget constraints'.",
    "Should resolve the lead and call crm_update on CRM Lead setting status to a lost/disqualified state and a reason field if one exists; should not drop the reason silently.")
add("crm_update", "Change the email address on Tom Bennett's contact to tbennett@newdomain.com.",
    "Should resolve the contact and call crm_update on Contact with the corrected email, confirming success.")
add("crm_update", "Update the Pinehill Labs organization's employee count to 250.",
    "Should resolve the organization and call crm_update with the employee count field set to 250, confirming success.")
add("crm_update", "Bump the Northwind Logistics deal's expected close date to end of next month.",
    "Should resolve the deal and call crm_update with a close date resolved to the actual last day of next month; should state the resolved date rather than leaving it ambiguous.")

# ---------------------------------------------------------------------------
# 5. crm_delete (8)
# ---------------------------------------------------------------------------
add("crm_delete", "Delete the task 'Schedule demo call'.",
    "Should resolve the task record, then call crm_delete on CRM Task. Given delete is destructive, it's acceptable (and preferable) for the agent to confirm with the user before deleting rather than deleting silently on the first message; either behavior should be judged on whether it avoids an unconfirmed destructive action without ever having surfaced what it's about to delete.")
add("crm_delete", "Remove the duplicate lead for Lena Wu.",
    "Should first identify which lead record is the duplicate (via search/get) before calling crm_delete on CRM Lead; should not delete the wrong or only existing record without first showing what it found, since 'duplicate' implies there are at least two candidates to distinguish between.")
add("crm_delete", "Delete the note about renewal terms on the Blue Harbor Retail deal.",
    "Should locate the specific note (there may be more than one matching 'renewal terms') and call crm_delete on CRM Note only after identifying the correct one; should not guess which note if multiple match.")
add("crm_delete", "Get rid of the call log entry from this morning with Sara Kim.",
    "Should resolve the specific call log record via search before calling crm_delete on CRM Call Log; should not delete based on a vague time reference without confirming which record matches.")
add("crm_delete", "Delete the organization 'Reef Analytics', it was created by mistake.",
    "Should resolve the organization and call crm_delete on CRM Organization; given deleting an organization may cascade or fail due to linked records, should surface any error from the tool rather than claiming success if the delete failed.")
add("crm_delete", "Please delete all leads with status 'Junk'.",
    "This is a bulk destructive request. The agent should either confirm the list of matching records before deleting each one, or clearly state it is proceeding to delete N records and report the outcome per record; should not claim a bulk delete succeeded without evidence from the tool calls actually made.")
add("crm_delete", "Remove the task assigned to me about the quarterly review deck.",
    "Should resolve the specific task (the agent doesn't inherently know who 'me' is, so it should ask or use known session identity) before calling crm_delete on CRM Task; should not delete an unrelated task.")
add("crm_delete", "Delete contact Tom Bennett.",
    "Should resolve the contact and call crm_delete on Contact; since this is irreversible and Tom Bennett could be linked to deals/organizations, a good answer either confirms first or clearly reports what happened (success, or an error if the record is linked elsewhere) rather than a vague 'done'.")

# ---------------------------------------------------------------------------
# 6. crm_linked_records (8)
# ---------------------------------------------------------------------------
add("crm_linked_records", "What records are linked to the Northwind Logistics organization?",
    "Should resolve the organization and call crm_linked_records, then summarize exactly what's returned (deals, contacts, etc.) without adding anything not present in the result.")
add("crm_linked_records", "Show me everything connected to the Atlas Manufacturing deal.",
    "Should resolve the deal and call crm_linked_records on CRM Deal, reporting only the linked records actually returned.")
add("crm_linked_records", "Which contacts are tied to Cedar & Co?",
    "Should resolve the organization and call crm_linked_records, then filter/report only the contact-type links from the result.")
add("crm_linked_records", "Does the Vertex Health deal have any linked tasks?",
    "Should call crm_linked_records on the Vertex Health deal and report whether any CRM Task links are present, based only on the actual tool output.")
add("crm_linked_records", "List all activity and linked records for Marigold Textiles.",
    "Likely needs both crm_activities and crm_linked_records for the organization; should report results from each and not conflate the two into a single fabricated list.")
add("crm_linked_records", "What's linked to Rahul Mehta's contact record?",
    "Should resolve the contact and call crm_linked_records, reporting exactly what comes back.")
add("crm_linked_records", "Is the Quantum Foods deal linked to any notes?",
    "Should call crm_linked_records on the Quantum Foods deal and report note-type links only if actually present in the result.")
add("crm_linked_records", "Show related records for the lead Kevin Ortiz.",
    "Should resolve the lead and call crm_linked_records, reporting only what the tool returns.")

# ---------------------------------------------------------------------------
# 7. crm_activities (8)
# ---------------------------------------------------------------------------
add("crm_activities", "What's the recent activity history on the Northwind Logistics deal?",
    "Should resolve the deal and call crm_activities, then summarize only the events actually returned, in a sensible order (e.g. most recent first) if timestamps are present.")
add("crm_activities", "Has anything happened on the Blue Harbor Retail deal in the last week?",
    "Should call crm_activities on the deal and filter/summarize to recent entries; should say clearly if there's no activity rather than inventing some.")
add("crm_activities", "Show me the activity timeline for lead Priya Nair.",
    "Should resolve the lead and call crm_activities, reporting only actual returned events.")
add("crm_activities", "What was the last interaction logged with Fatima Al-Sayed?",
    "Should resolve the contact/related record and call crm_activities, then report the most recent entry actually returned, not a guess.")
add("crm_activities", "Summarize all activity on the Atlas Manufacturing deal so far.",
    "Should call crm_activities on the deal and produce a summary grounded only in the returned events.")
add("crm_activities", "Has the Reef Analytics organization had any recent updates?",
    "Should resolve the organization and call crm_activities (and/or crm_linked_records), reporting actual findings rather than assuming activity exists.")
add("crm_activities", "What actions were taken on the Quantum Foods deal this month?",
    "Should call crm_activities on the deal and filter to the current month, reporting only what's actually present.")
add("crm_activities", "Give me a timeline of everything that's happened with the Vertex Health deal.",
    "Should call crm_activities on the deal and present a chronological summary grounded in the actual tool output.")

# ---------------------------------------------------------------------------
# 8. crm_contact_action (6)
# ---------------------------------------------------------------------------
add("crm_contact_action", "Mark Rahul Mehta as the primary contact for his organization.",
    "Should resolve the contact and call crm_contact_action with an action that sets the primary-contact field/flag, confirming success.")
add("crm_contact_action", "Set Sara Kim's email as her primary email address.",
    "Should call crm_contact_action targeting the contact's email with the appropriate action/field/value, confirming success.")
add("crm_contact_action", "Add a secondary phone number for Daniel Obi: +234-80-1111222.",
    "Should call crm_contact_action to add a phone entry to Daniel Obi's contact, confirming success.")
add("crm_contact_action", "Remove the old email address on Tom Bennett's contact.",
    "Should call crm_contact_action with a removal action on the specified email field/value, confirming success or reporting if the value wasn't found.")
add("crm_contact_action", "Make the new phone number on Maria Lopez's contact the primary one.",
    "Should call crm_contact_action setting the primary flag on the specified phone entry, confirming success.")
add("crm_contact_action", "Update the primary email on Lena Wu's contact to lena.wu@newmail.com.",
    "Should call crm_contact_action to set/update the primary email, confirming success; if the email doesn't already exist as an entry it should add it rather than failing silently.")

# ---------------------------------------------------------------------------
# 9. crm_research_company -- web research, no CRM writes (6)
# ---------------------------------------------------------------------------
add("crm_research_company", "Research the company Northwind Logistics and give me a quick brief before my call.",
    "Should call crm_research_company (web-based) and produce a brief grounded in what the tool returned; must not write anything to the CRM, and should note if information couldn't be found rather than fabricating a company profile.")
add("crm_research_company", "What can you find out about Atlas Manufacturing online?",
    "Should call crm_research_company and summarize actual findings; should not silently also create/update CRM records unless separately asked.")
add("crm_research_company", "Look up Cedar & Co and tell me what industry they're in and roughly how big they are.",
    "Should call crm_research_company and answer using only what the research turned up, flagging uncertainty if the source data is thin or ambiguous.")
add("crm_research_company", "I have a call with Vertex Health tomorrow -- can you brief me on who they are?",
    "Should call crm_research_company for Vertex Health and produce a concise pre-call brief grounded in the tool's findings, not general knowledge presented as fact.")
add("crm_research_company", "Find recent news about Quantum Foods.",
    "Should call crm_research_company (or the underlying web tool) and report only what was actually found, with no fabricated headlines or dates.")
add("crm_research_company", "Does Marigold Textiles have a public website, and what do they do?",
    "Should call crm_research_company and answer based on actual findings, being explicit if no public website could be located rather than inventing a URL.")

# ---------------------------------------------------------------------------
# 10. Multi-tool / reasoning / edge cases / out-of-scope (4)
# ---------------------------------------------------------------------------
add("multi_step", "Find the Northwind Logistics deal, tell me its status, and then create a follow-up task for next Monday.",
    "Requires chaining crm_search/crm_get (to find and read the deal) with crm_create (for the task), reporting the deal's actual status (Frappe CRM deals use a status field, not 'stage') and confirming the task was created with a correctly resolved date for 'next Monday'.")
add("edge_case", "Update the status of the lead 'Zzyzx Corp' to Qualified.",
    "The record almost certainly doesn't exist. Should attempt to resolve it (search/get), find no match, and clearly report that the lead wasn't found rather than fabricating a success message or inventing a record.")
add("edge_case", "Create a record in the 'Invoice' doctype for this deal.",
    "'Invoice' is not one of the doctypes this agent is permitted to touch (only CRM Organization, CRM Lead, CRM Deal, Contact, CRM Task, CRM Note, CRM Call Log, and a few read-only reference doctypes). Should decline or explain that this doctype isn't supported, rather than attempting the call or pretending it succeeded.")
add("out_of_scope", "What's the weather like in Mumbai today, and also can you pull up the Northwind Logistics deal?",
    "The weather portion is out of scope for a CRM agent with no weather tool; a good answer either says it can't check the weather or ignores that part gracefully, while still correctly handling the CRM portion (looking up the Northwind Logistics deal) without fabricating a weather report.")

assert len(rows) == 100, f"expected 100 questions, generated {len(rows)}"


def write_csv(fh):
    writer = csv.DictWriter(fh, fieldnames=["id", "category", "question", "rubric"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Write to this path instead of stdout")
    args = parser.parse_args()

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            write_csv(fh)
        print(f"Wrote {len(rows)} questions to {args.out}", file=sys.stderr)
    else:
        buf = io.StringIO()
        write_csv(buf)
        sys.stdout.write(buf.getvalue())


if __name__ == "__main__":
    main()