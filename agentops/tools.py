"""In memory mock Jira and ServiceNow backends plus a typed tool registry.

The mocks mimic the shape of the real APIs (issue keys like OPS-12, incident numbers like
INC0010001, workflow transitions) without any network access.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

JIRA_TRANSITIONS = {"To Do": {"In Progress"}, "In Progress": {"In Review", "To Do"}, "In Review": {"Done", "In Progress"}, "Done": set()}
SN_STATES = ["New", "In Progress", "On Hold", "Resolved", "Closed"]


@dataclass
class MockJira:
    project: str = "OPS"
    issues: dict[str, dict[str, Any]] = field(default_factory=dict)
    _seq: int = 0

    def create_issue(self, summary: str, description: str = "", priority: str = "Medium") -> dict[str, Any]:
        self._seq += 1
        key = f"{self.project}-{self._seq}"
        self.issues[key] = {"key": key, "summary": summary, "description": description, "priority": priority, "status": "To Do"}
        return self.issues[key]

    def search_issues(self, text: str = "", status: str | None = None) -> list[dict[str, Any]]:
        t = text.lower()
        return [i for i in self.issues.values() if (t in i["summary"].lower() or t in i["description"].lower()) and (status is None or i["status"] == status)]

    def transition_issue(self, key: str, to_status: str) -> dict[str, Any]:
        issue = self.issues.get(key)
        if issue is None:
            raise KeyError(f"issue {key} not found")
        if to_status not in JIRA_TRANSITIONS[issue["status"]]:
            raise ValueError(f"cannot move {key} from {issue['status']} to {to_status}")
        issue["status"] = to_status
        return issue


@dataclass
class MockServiceNow:
    incidents: dict[str, dict[str, Any]] = field(default_factory=dict)
    _seq: int = 10000

    def create_incident(self, short_description: str, urgency: int = 3) -> dict[str, Any]:
        self._seq += 1
        number = f"INC00{self._seq}"
        self.incidents[number] = {"number": number, "short_description": short_description, "urgency": urgency, "state": "New", "work_notes": []}
        return self.incidents[number]

    def get_incident(self, number: str) -> dict[str, Any]:
        if number not in self.incidents:
            raise KeyError(f"incident {number} not found")
        return self.incidents[number]

    def update_incident(self, number: str, state: str | None = None, work_note: str | None = None) -> dict[str, Any]:
        inc = self.get_incident(number)
        if state is not None:
            if state not in SN_STATES:
                raise ValueError(f"unknown state {state}")
            inc["state"] = state
        if work_note:
            inc["work_notes"].append(work_note)
        return inc


# Typed argument schemas: the agent's tool calls are validated before execution.
class CreateIssueArgs(BaseModel):
    summary: str = Field(min_length=3, max_length=200)
    description: str = Field(default="", max_length=2000)
    priority: Literal["Low", "Medium", "High", "Critical"] = "Medium"


class SearchIssuesArgs(BaseModel):
    text: str = Field(default="", max_length=200)
    status: str | None = None


class TransitionIssueArgs(BaseModel):
    key: str = Field(pattern=r"^[A-Z]+-\d+$")
    to_status: Literal["To Do", "In Progress", "In Review", "Done"]


class CreateIncidentArgs(BaseModel):
    short_description: str = Field(min_length=3, max_length=160)
    urgency: int = Field(default=3, ge=1, le=3)


class GetIncidentArgs(BaseModel):
    number: str = Field(pattern=r"^INC\d{7}$")


class UpdateIncidentArgs(BaseModel):
    number: str = Field(pattern=r"^INC\d{7}$")
    state: str | None = None
    work_note: str | None = Field(default=None, max_length=1000)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]
    side_effect: bool


class ToolError(Exception):
    pass


class ToolRegistry:
    def __init__(self, jira: MockJira | None = None, snow: MockServiceNow | None = None):
        self.jira = jira or MockJira()
        self.snow = snow or MockServiceNow()
        self.tools: dict[str, Tool] = {
            t.name: t
            for t in [
                Tool("jira_create_issue", "Create a Jira issue", CreateIssueArgs, self.jira.create_issue, True),
                Tool("jira_search_issues", "Search Jira issues", SearchIssuesArgs, self.jira.search_issues, False),
                Tool("jira_transition_issue", "Move a Jira issue to a new status", TransitionIssueArgs, self.jira.transition_issue, True),
                Tool("snow_create_incident", "Open a ServiceNow incident", CreateIncidentArgs, self.snow.create_incident, True),
                Tool("snow_get_incident", "Read a ServiceNow incident", GetIncidentArgs, self.snow.get_incident, False),
                Tool("snow_update_incident", "Update a ServiceNow incident", UpdateIncidentArgs, self.snow.update_incident, True),
            ]
        }

    def call(self, name: str, args: dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name}")
        try:
            parsed = tool.args_model(**args)
        except ValidationError as e:
            raise ToolError(f"invalid arguments for {name}: {e.errors()[0]['msg']}") from e
        try:
            return tool.fn(**parsed.model_dump())
        except (KeyError, ValueError) as e:
            raise ToolError(str(e).strip("'\"")) from e
