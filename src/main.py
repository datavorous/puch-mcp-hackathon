from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from typing import List, Annotated, Optional, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from mcp.types import INVALID_PARAMS
from mcp import ErrorData, McpError
import asyncpraw, yaml, asyncio, os
from dotenv import load_dotenv
from intent import get_intent
from fastmcp import FastMCP
from rapidfuzz import fuzz
from time import time
import json
import threading
import asyncio
import io, httpx
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import base64
from typing_extensions import Annotated
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR


load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

DATA_FILE = "user_data.json"
USER_DATA = {}
USER_ACTIVITY_CACHE = {}
USER_AGENT = os.environ.get("USER_AGENT")

COOLDOWN_SECONDS = 80

file_lock = threading.Lock()


def init_data():
    global USER_DATA
    if not os.path.exists(DATA_FILE):
        USER_DATA = {}
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        USER_DATA = json.loads(content) if content.strip() else {}
    except (json.JSONDecodeError, OSError) as e:
        print("Failed to load data:", e)
        USER_DATA = {}


def save_data():
    tmp = DATA_FILE + ".tmp"
    try:
        with file_lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(USER_DATA, f, indent=4)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp, DATA_FILE)
        print(f"Data saved to {DATA_FILE}")
    except Exception as e:
        print("Failed to save data:", e)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def cooldown_check(user_id: str, tool_name: str):
    current_time = time()

    if user_id not in USER_ACTIVITY_CACHE:
        USER_ACTIVITY_CACHE[user_id] = {}

    user_cache = USER_ACTIVITY_CACHE[user_id]
    last_used = user_cache.get(tool_name, 0)
    time_elapsed = current_time - last_used

    if time_elapsed >= COOLDOWN_SECONDS:
        user_cache[tool_name] = current_time
        return True, 0.0
    else:
        remaining = COOLDOWN_SECONDS - time_elapsed
        return False, remaining


class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


mcp = FastMCP(
    "MCPs for PuchAI Hackathon",
    auth=SimpleBearerAuthProvider(TOKEN),
)


# REDDIT SCRAPER TOOL

RedditRecentPostsScraper = RichToolDescription(
    description="Scrapes latest new posts from specified subreddits and filters down posts which contain specific keywords, before analysing the content.",
    use_when="Use this when you need to find 'recent' reddit posts containing specific keywords across multiple subreddits, and tell the user which is the most engaging using intent scoring. Perform each subreddit search in sequential order to avoid time outs. Once a single subreddit is done, move on to the next one. Only meant for 'new' posts, not 'hot' or 'top'. Highly helpful for market validation and audience sentiment analysis.",
    side_effects="Fetches data from Reddit API.",
)


@mcp.tool(description=RedditRecentPostsScraper.model_dump_json())
async def reddit_recent_posts_scraper(
    puch_user_id: Annotated[str, Field(description="User ID for rate limiting")],
    query: Annotated[
        str, Field(description="A descriptive query about what you're looking for")
    ],
    subreddits: Annotated[
        List[str], Field(description="List of subreddit names (without 'r/' prefix)")
    ],
    keywords: Annotated[
        List[str],
        Field(description="List of keywords to search for using fuzzy matching"),
    ],
    threshold: Annotated[
        int, Field(description="Fuzzy match threshold (0-100, default: 80)", default=80)
    ],
    post_limit: Annotated[
        int,
        Field(
            description="Number of new posts to check per subreddit (default: 10)",
            default=10,
        ),
    ],
) -> str:
    if not puch_user_id.strip():
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message="User ID cannot be empty.")
        )
    if not query.strip():
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Query cannot be empty."))
    if not subreddits:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS, message="At least one subreddit must be specified."
            )
        )
    if not keywords:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS, message="At least one keyword must be specified."
            )
        )
    if not (0 <= threshold <= 100):
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS, message="Threshold must be between 0 and 100."
            )
        )
    if post_limit < 1 or post_limit > 35:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS, message="Post limit must be between 1 and 35."
            )
        )

    can_use, remaining_time = cooldown_check(puch_user_id, "reddit_scraper")
    if not can_use:
        remaining_minutes = remaining_time / 60
        return f"Tell the user that Reddit Scraper is on cooldown. Please wait {remaining_minutes} minutes before using again."

    async with asyncpraw.Reddit(
        client_id=os.environ.get("CLIENT_ID"),
        client_secret=os.environ.get("CLIENT_SECRET"),
        user_agent=os.environ.get("USER_AGENT_REDDIT"),
        read_only=True,
    ) as reddit:
        try:
            results = []
            total_matches = 0

            for subreddit_name in subreddits:
                print(f"Processing subreddit: {subreddit_name}")
                subreddit_matches = []

                try:
                    try:
                        subreddit = await reddit.subreddit(subreddit_name, fetch=True)
                    except Exception as e:
                        print(f"Error fetching subreddit {subreddit_name}: {str(e)}")
                        results.append(
                            {
                                "subreddit": subreddit_name,
                                "error": f"Could not access r/{subreddit_name}: {str(e)}",
                            }
                        )
                        continue
                    post_i_saw = 0
                    async for post in subreddit.new(limit=post_limit):
                        post_i_saw += 1
                        post_text = (post.title or "") + "\n" + (post.selftext or "")

                        # comments are not really needed for this task, so skipping fetching comments
                        # try:
                        # comments = await post.comments()
                        # comments.replace_more(limit=0)
                        # not loading "more comments" to avoid rate limits
                        # comment_text = ""
                        # for comment in comments.list():
                        #  if getattr(comment, "body", None):
                        #     comment_text += "\n" + comment.body

                        # full_text = post_text + comment_text
                        # pass
                        # except Exception as e:
                        #     print(f"Error fetching comments for post {post.id}: {str(e)}")

                        full_text = post_text

                        matched_keywords = []
                        for keyword in keywords:
                            score = fuzz.token_set_ratio(
                                keyword.lower(), full_text.lower()
                            )
                            if score >= threshold:
                                matched_keywords.append((keyword, score))
                        if matched_keywords:
                            author_karma = 0
                            if post.author:
                                try:
                                    redditor = await reddit.redditor(
                                        post.author.name, fetch=True
                                    )
                                    author_karma = getattr(redditor, "link_karma", 0)
                                except Exception as e:
                                    print(
                                        f"Could not fetch redditor {post.author.name}: {e}"
                                    )
                                    author_karma = 0
                            match_info = {
                                "title": post.title,
                                "url": f"https://reddit.com{post.permalink}",
                                "score": post.score,
                                "num_comments": post.num_comments,
                                "subreddit": subreddit_name,
                                "matched_keywords": matched_keywords,
                                "full_text": post.selftext[:600]
                                .replace("\n", " ")
                                .replace("  ", "")
                                + (
                                    "..."
                                    if len(post.selftext) > 600
                                    else "No Post Text"
                                ),
                                "intent_score": get_intent(
                                    upvotes=post.score,
                                    comments=post.num_comments,
                                    ratio=post.upvote_ratio,
                                    created_utc=post.created_utc,
                                    current_utc=datetime.now(timezone.utc).timestamp(),
                                    author_karma=author_karma,
                                    keyword_matches=len(matched_keywords),
                                ),
                            }
                            subreddit_matches.append(match_info)
                            total_matches += 1
                            # print(f"TOTAL MATCHES: {total_matches}")

                    if subreddit_matches:
                        results.append(
                            {"subreddit": subreddit_name, "matches": subreddit_matches}
                        )

                except Exception as e:
                    results.append(
                        {
                            "subreddit": subreddit_name,
                            "error": f"Error accessing r/{subreddit_name}: {str(e)}",
                        }
                    )

            if total_matches == 0:
                print("No MATCHES FOUND")
                return f"No matches found\n\nTried searching {post_limit} recent posts from each subreddit."

            output = f"Reddit Keyword Search Results\n\nQuery: {query}\nKeywords: {', '.join(keywords)}\nTotal Matches: {total_matches}\n\n"

            for result in results:
                if "error" in result:
                    output += f"r/{result['subreddit']}: {result['error']}\n\n"
                    continue

                output += (
                    f"r/{result['subreddit']} ({len(result['matches'])} matches)\n\n"
                )

            all_matches = []
            for result in results:
                if "matches" in result:
                    all_matches.extend(result["matches"])

            # sorting by intent score
            all_matches.sort(key=lambda x: x["intent_score"], reverse=True)

            # perhaps yaml would be better than a json prompt?
            output_data = []
            for i, match in enumerate(all_matches, 1):
                intent_level = "LOW"
                if match["intent_score"] > 10:
                    intent_level = "HIGH"
                elif match["intent_score"] > 5:
                    intent_level = "MEDIUM"

                output_data.append(
                    {
                        "index": i,
                        "title": match["title"][:80]
                        + ("..." if len(match["title"]) > 80 else ""),
                        "content": match["full_text"],
                        "url": match["url"],
                        "upvotes": match["score"],
                        "comments": match["num_comments"],
                        "subreddit": match["subreddit"],
                        "intent": {
                            "level": intent_level,
                            "score": round(match["intent_score"], 2),
                        },
                    }
                )

            output += yaml.dump(output_data, sort_keys=False, allow_unicode=True)
            output += "USE THIS DATA TO:\n"
            output += "- Detect emotional weak points in audience sentiment.\n"
            output += "- MENTION THE URLS/LINKS to THE POST ALWAYS.\n"
            output += "- Suggest how to reply to the posts whilst following Machiavellian principles.\n\n"

            # print(output)
            print(
                f"Total Matches Found: {total_matches} in {len(subreddits)} subreddits"
            )
            return output

        except Exception as e:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message=f"Reddit API error: {str(e)}")
            )


# TODO LIST MANAGEMENT TOOLS HERE

todo_add = RichToolDescription(
    description="A tool to add a todo item for the user.",
    use_when="Use this when the user wants to add a task to their TODO list.",
    side_effects="Saves the new todo item to a JSON file.",
)


@mcp.tool(description=todo_add.model_dump_json())
async def add_todo(
    puch_user_id: Annotated[str, Field(description="User ID")],
    task: Annotated[str, Field(description="The task or todo item text")],
) -> str:
    user_data = USER_DATA.setdefault(puch_user_id, {"todo": [], "preferences": {}})
    user_data["todo"].append(
        {"task": task, "created_at": datetime.now(timezone.utc).isoformat()}
    )
    await asyncio.to_thread(save_data)
    print(f"TODO added for user {puch_user_id}: {task}")
    return f"Your todo '{task}' has been added."


todo_list = RichToolDescription(
    description="A tool to list all TODO items for a user.",
    use_when="Use this to retrieve a list of tasks in the user's TODO list.",
    side_effects="Reads todo data from a JSON file.",
)


@mcp.tool(description=todo_list.model_dump_json())
async def list_todos(puch_user_id: Annotated[str, Field(description="User ID")]) -> str:
    user_data = USER_DATA.get(puch_user_id, {})
    todos = user_data.get("todo", [])

    if not todos:
        return "No todos found. Tell the user that no todos were found"
    todos_sorted = sorted(todos, key=lambda t: t.get("created_at", ""))

    output_lines = []
    for idx, todo in enumerate(todos_sorted):
        created_at_str = todo.get("created_at")
        if created_at_str:
            created_at_dt = datetime.fromisoformat(created_at_str)
            created_at_fmt = created_at_dt.strftime("%Y-%m-%d %H:%M UTC")
        else:
            created_at_fmt = "Unknown date"

        output_lines.append(f"{idx+1}. {todo['task']} (added on {created_at_fmt})")

    output = "\n".join(output_lines)
    output += "\n\nThese are your current TODO items. Show these to the User in the form of a list."
    print(output)
    return output


todo_delete = RichToolDescription(
    description="A tool to delete a todo item by its index.",
    use_when="Use this to delete a specific task from the user's TODO list.",
    side_effects="Removes the todo from the JSON file.",
)


@mcp.tool(description=todo_delete.model_dump_json())
async def delete_todo(
    puch_user_id: Annotated[str, Field(description="User ID")],
    index: Annotated[
        int, Field(description="The index number of the todo from list_todos (1-based)")
    ],
) -> str:
    user_data = USER_DATA.get(puch_user_id, {})
    todos = user_data.get("todo", [])

    if not todos:
        return "No todos to delete."

    if index < 1 or index > len(todos):
        return f"Invalid index. Please choose between 1 and {len(todos)}."

    removed_task = todos.pop(index - 1)["task"]
    await asyncio.to_thread(save_data)
    return f"Todo '{removed_task}' deleted."


# USER PREFERENCES

pref_save_desc = RichToolDescription(
    description="A tool to save user preferences, likes, interests etc. into preferences storage.",
    use_when="Use this when the you detect something memorable or important about user's likes/interests/preferences/life choices.",
    side_effects="Updates the 'preferences' dictionary for the user in the JSON data.",
)

pref_get_desc = RichToolDescription(
    description="Retrieve user preferences or a specific preference value by key.",
    use_when="Use this when the AI needs to recall something memorable or important about user's likes/interests/preferences/life choices.",
    side_effects="Reads preferences from the JSON file.",
)


@mcp.tool(description=pref_save_desc.model_dump_json())
async def save_user_preference(
    puch_user_id: Annotated[str, Field(description="User ID")],
    key: Annotated[
        str, Field(description="Preference category or key (e.g., 'favorite_color')")
    ],
    value: Annotated[str, Field(description="Value to save under the given key")],
) -> str:
    user_data = USER_DATA.setdefault(
        puch_user_id,
        {"memos": [], "todo": [], "preferences": {}, "short_term_memory": []},
    )
    user_data["preferences"][key] = value
    await asyncio.to_thread(save_data)
    print(f"Preference saved for user {puch_user_id}: {key} = {value}")
    return f"Preference '{key}' saved."


@mcp.tool(description=pref_get_desc.model_dump_json())
async def get_user_preference(
    puch_user_id: Annotated[str, Field(description="User ID")],
    key: Optional[
        Annotated[
            str, Field(description="Specific preference key to retrieve (optional)")
        ]
    ] = None,
) -> str:
    print("trid")
    user_data = USER_DATA.get(puch_user_id, {})
    preferences = user_data.get("preferences", {})
    if not preferences:
        return "No preferences found for the user."

    if key:
        print("key got")
        value = preferences.get(key)
        if value is None:
            return f"No preference found for key '{key}'."
        return f"Preference '{key}': {value}"

    output_lines = [f"{k}: {v}" for k, v in preferences.items()]
    return "User preferences:\n" + "\n".join(output_lines)


######## GRAPH PLOTTING ######## ---- ######## ----- ############
# DO NOT TOUCH


make_bar_chart = RichToolDescription(
    description="Generate a labeled bar chart image from lists of labels and numeric data.",
    use_when="Use when you want a quick visual (PNG) representing the provided labels and values.",
    side_effects="No persistent side effects; returns an image to the caller.",
)


@mcp.tool(description=make_bar_chart.model_dump_json())
async def make_labeled_bar_chart(
    puch_labels: Annotated[
        List[str], Field(description="List of label strings (x-axis).")
    ],
    puch_values: Annotated[
        List[float],
        Field(description="List of numeric values (same length as labels)."),
    ],
    width: Annotated[int, Field(description="Image width in px (optional)")] = 900,
    height: Annotated[int, Field(description="Image height in px (optional)")] = 450,
    bg_color: Annotated[
        Optional[str], Field(description="Background color, e.g. 'white'")
    ] = "#f8fffe",
    bar_color: Annotated[
        Optional[str], Field(description="Bar color, e.g. 'steelblue'")
    ] = "#4ade80",
    title: Annotated[Optional[str], Field(description="Optional chart title")] = None,
) -> list[TextContent | ImageContent]:
    try:
        if puch_labels is None or puch_values is None:
            raise ValueError("Both 'labels' and 'data' must be provided.")

        labels = list(puch_labels)
        values = list(puch_values)

        if len(labels) != len(values):
            raise ValueError(
                "Length mismatch: 'labels' and 'data' must have the same length."
            )
        if len(labels) == 0:
            raise ValueError(
                "Empty lists provided. Provide at least one label/value pair."
            )

        MAX_BARS = 80
        if len(labels) > MAX_BARS:
            raise ValueError(
                f"Too many bars ({len(labels)}). Max supported is {MAX_BARS}."
            )

        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except Exception:
                raise ValueError(f"Non-numeric data value found: {v}")

        def _get_darker_variant(color: str) -> str:
            if color.startswith("#") and len(color) == 7:
                try:
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)

                    r = max(0, int(r * 0.6))
                    g = max(0, int(g * 0.6))
                    b = max(0, int(b * 0.6))

                    return f"#{r:02x}{g:02x}{b:02x}"
                except:
                    pass

            color_map = {
                "#4ade80": "#16a34a",
                "steelblue": "#4682b4",
                "red": "#8b0000",
                "blue": "#000080",
                "orange": "#ff4500",
                "purple": "#4b0082",
            }
            return color_map.get(color.lower(), "#2d2d2d")

        def _measure_text(
            draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
        ):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return w, h

        def _truncate_to_width(
            draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int
        ):
            if max_w <= 0:
                return ""
            w, _ = _measure_text(draw, text, font)
            if w <= max_w:
                return text
            if len(text) <= 1:
                return text[:1]
            truncated = text

            ellipsis = "…"
            while len(truncated) > 0:
                truncated = truncated[:-1]
                candidate = truncated + ellipsis
                cw, _ = _measure_text(draw, candidate, font)
                if cw <= max_w:
                    return candidate
            return ellipsis

        def _create_chart_image(labels: List[str], values: List[float]) -> bytes:
            pad_x = 80
            pad_y = 80
            title_space = 60 if title else 30

            img_w = max(200, width)
            img_h = max(120, height)

            chart_w = img_w - pad_x * 2
            chart_h = img_h - pad_y * 2 - title_space

            img = Image.new("RGB", (img_w, img_h), color=bg_color)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("gsans.ttf", 20)
                title_font = ImageFont.truetype("gsans.ttf", 24)
            except:
                print("the fonts were not present")
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                    title_font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                    title_font = ImageFont.load_default()

            max_v = max(values)
            top_value = max_v if max_v > 0 else 1.0
            if top_value == 0:
                top_value = 1.0

            n = len(values)
            spacing = max(12, int(chart_w * 0.03))
            total_spacing = spacing * (n + 1)
            bar_total_space = max(0, chart_w - total_spacing)
            bar_w = max(8, int(bar_total_space / n)) if n > 0 else 8

            grid_color = "#e5e7eb"
            ticks = 5
            for i in range(ticks + 1):
                y_value = top_value * (i / ticks)
                y = pad_y + title_space + int(chart_h * (1 - i / ticks))

                if i > 0:
                    draw.line(
                        [(pad_x, y), (pad_x + chart_w, y)], fill=grid_color, width=1
                    )

                label_text = f"{y_value:.1f}" if y_value < 10 else f"{y_value:.0f}"
                lbl_w, lbl_h = _measure_text(draw, label_text, font)
                draw.text(
                    (pad_x - 12 - lbl_w, y - lbl_h / 2),
                    label_text,
                    font=font,
                    fill="#6b7280",
                )

            x0 = pad_x
            y0 = pad_y + title_space
            x1 = pad_x + chart_w
            y1 = pad_y + title_space + chart_h

            draw.line([(x0, y1), (x1, y1)], fill="#9ca3af", width=2)
            draw.line([(x0, y0), (x0, y1)], fill="#9ca3af", width=2)

            border_color = _get_darker_variant(bar_color)

            for idx, (lbl, val) in enumerate(zip(labels, values)):
                bx = pad_x + spacing * (idx + 1) + bar_w * idx
                h_frac = (val / top_value) if top_value != 0 else 0
                bh = int(chart_h * h_frac)
                by = y1 - bh

                corner_radius = min(4, bar_w // 4)

                draw.rectangle(
                    [bx, by, bx + bar_w, y1],
                    fill=bar_color,
                    outline=border_color,
                    width=2,
                )

                if bh > corner_radius * 2:
                    for i in range(corner_radius):
                        offset = corner_radius - i - 1
                        draw.rectangle(
                            [bx + offset, by + i, bx + bar_w - offset, by + i + 1],
                            fill=bar_color,
                        )

                max_label_w = bar_w + spacing - 4
                truncated_label = _truncate_to_width(draw, str(lbl), font, max_label_w)
                text_w, text_h = _measure_text(draw, truncated_label, font)
                tx = bx + (bar_w - text_w) / 2
                ty = y1 + 8
                draw.text((tx, ty), truncated_label, font=font, fill="#374151")

                val_text = f"{val:.1f}" if val < 10 else f"{val:.0f}"
                tw, th = _measure_text(draw, val_text, font)
                vx = bx + (bar_w - tw) / 2
                vy = max(by - th - 6, pad_y + 4)

                draw.rectangle(
                    [vx - 2, vy - 1, vx + tw + 2, vy + th + 1],
                    fill="#ffffff",
                    outline=None,
                )
                draw.text((vx, vy), val_text, font=font, fill="#1f2937")

            if title:
                t_w, t_h = _measure_text(draw, title, title_font)
                draw.text(
                    ((img_w - t_w) / 2, 20), title, font=title_font, fill="#111827"
                )

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        png_bytes = await asyncio.to_thread(_create_chart_image, labels, numeric_values)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return [ImageContent(type="image", mimeType="image/png", data=b64)]

    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# LINE CHART PLOTTING TOOL HERE


make_line_chart = RichToolDescription(
    description="Generate a labeled line chart image from lists of labels and numeric data.",
    use_when="Use when you want a quick visual (PNG) representing the provided labels and values as a line chart.",
    side_effects="No persistent side effects; returns an image to the caller.",
)


@mcp.tool(description=make_line_chart.model_dump_json())
async def make_labeled_line_chart(
    puch_labels: Annotated[
        List[str], Field(description="List of label strings (x-axis).")
    ],
    puch_values: Annotated[
        List[float],
        Field(description="List of numeric values (same length as labels)."),
    ],
    width: Annotated[int, Field(description="Image width in px (optional)")] = 900,
    height: Annotated[int, Field(description="Image height in px (optional)")] = 450,
    bg_color: Annotated[
        Optional[str], Field(description="Background color, e.g. 'white'")
    ] = "#f8fffe",
    line_color: Annotated[
        Optional[str], Field(description="Line color, e.g. 'steelblue'")
    ] = "#4ade80",
    title: Annotated[Optional[str], Field(description="Optional chart title")] = None,
) -> list[TextContent | ImageContent]:
    try:
        if puch_labels is None or puch_values is None:
            raise ValueError("Both 'labels' and 'data' must be provided.")

        labels = list(puch_labels)
        values = list(puch_values)

        if len(labels) != len(values):
            raise ValueError(
                "Length mismatch: 'labels' and 'data' must have the same length."
            )
        if len(labels) == 0:
            raise ValueError(
                "Empty lists provided. Provide at least one label/value pair."
            )

        MAX_POINTS = 200
        if len(labels) > MAX_POINTS:
            raise ValueError(
                f"Too many points ({len(labels)}). Max supported is {MAX_POINTS}."
            )

        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except Exception:
                raise ValueError(f"Non-numeric data value found: {v}")

        def _get_darker_variant(color: str) -> str:
            """Get a darker variant of any color for point borders"""
            if color.startswith("#") and len(color) == 7:
                try:
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)

                    r = max(0, int(r * 0.6))
                    g = max(0, int(g * 0.6))
                    b = max(0, int(b * 0.6))

                    return f"#{r:02x}{g:02x}{b:02x}"
                except:
                    pass

            color_map = {
                "#4ade80": "#16a34a",
                "steelblue": "#4682b4",
                "red": "#8b0000",
                "blue": "#000080",
                "orange": "#ff4500",
                "purple": "#4b0082",
            }
            return color_map.get(color.lower(), "#2d2d2d")

        def _measure_text(
            draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
        ):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return w, h

        def _truncate_to_width(
            draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int
        ):
            if max_w <= 0:
                return ""
            w, _ = _measure_text(draw, text, font)
            if w <= max_w:
                return text
            if len(text) <= 1:
                return text[:1]
            truncated = text

            ellipsis = "…"
            while len(truncated) > 0:
                truncated = truncated[:-1]
                candidate = truncated + ellipsis
                cw, _ = _measure_text(draw, candidate, font)
                if cw <= max_w:
                    return candidate
            return ellipsis

        def _create_chart_image(labels: List[str], values: List[float]) -> bytes:
            pad_x = 80
            pad_y = 80
            title_space = 60 if title else 30

            img_w = max(200, width)
            img_h = max(120, height)

            chart_w = img_w - pad_x * 2
            chart_h = img_h - pad_y * 2 - title_space

            img = Image.new("RGB", (img_w, img_h), color=bg_color)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("gsans.ttf", 20)
                title_font = ImageFont.truetype("gsans.ttf", 24)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                    title_font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                    title_font = ImageFont.load_default()

            min_v = min(values)
            max_v = max(values)

            range_v = max_v - min_v
            if range_v == 0:
                range_v = 1.0
                min_v = max_v - 0.5
                max_v = max_v + 0.5
            else:
                padding = range_v * 0.1
                min_v = min_v - padding
                max_v = max_v + padding

            n = len(values)

            grid_color = "#e5e7eb"
            ticks = 5
            for i in range(ticks + 1):
                y_value = min_v + (max_v - min_v) * (i / ticks)
                y = pad_y + title_space + int(chart_h * (1 - i / ticks))

                if i > 0 and i < ticks:
                    draw.line(
                        [(pad_x, y), (pad_x + chart_w, y)], fill=grid_color, width=1
                    )

                label_text = f"{y_value:.1f}" if abs(y_value) < 10 else f"{y_value:.0f}"
                lbl_w, lbl_h = _measure_text(draw, label_text, font)
                draw.text(
                    (pad_x - 12 - lbl_w, y - lbl_h / 2),
                    label_text,
                    font=font,
                    fill="#6b7280",
                )

            x0 = pad_x
            y0 = pad_y + title_space
            x1 = pad_x + chart_w
            y1 = pad_y + title_space + chart_h

            draw.line([(x0, y1), (x1, y1)], fill="#9ca3af", width=2)
            draw.line([(x0, y0), (x0, y1)], fill="#9ca3af", width=2)

            points = []
            point_radius = 4
            darker_color = _get_darker_variant(line_color)

            for idx, (lbl, val) in enumerate(zip(labels, values)):
                px = pad_x + (chart_w * idx / (n - 1)) if n > 1 else pad_x + chart_w / 2

                if max_v != min_v:
                    py = y1 - int(chart_h * (val - min_v) / (max_v - min_v))
                else:
                    py = y1 - chart_h / 2

                points.append((px, py))

                max_label_w = chart_w // (n + 1) if n > 1 else chart_w
                truncated_label = _truncate_to_width(draw, str(lbl), font, max_label_w)
                text_w, text_h = _measure_text(draw, truncated_label, font)
                tx = px - text_w / 2
                ty = y1 + 8
                draw.text((tx, ty), truncated_label, font=font, fill="#374151")

            if len(points) > 1:
                draw.line(points, fill=line_color, width=3)

            for idx, ((px, py), val) in enumerate(zip(points, values)):
                draw.ellipse(
                    [
                        px - point_radius,
                        py - point_radius,
                        px + point_radius,
                        py + point_radius,
                    ],
                    fill=line_color,
                    outline=darker_color,
                    width=2,
                )

                val_text = f"{val:.1f}" if abs(val) < 10 else f"{val:.0f}"
                tw, th = _measure_text(draw, val_text, font)
                vx = px - tw / 2
                vy = max(py - th - point_radius - 8, pad_y + 4)

                draw.rectangle(
                    [vx - 2, vy - 1, vx + tw + 2, vy + th + 1],
                    fill="#ffffff",
                    outline=None,
                )
                draw.text((vx, vy), val_text, font=font, fill="#1f2937")

            if title:
                t_w, t_h = _measure_text(draw, title, title_font)
                draw.text(
                    ((img_w - t_w) / 2, 20), title, font=title_font, fill="#111827"
                )

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        png_bytes = await asyncio.to_thread(_create_chart_image, labels, numeric_values)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return [ImageContent(type="image", mimeType="image/png", data=b64)]

    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# PIE CHART PLOTTING TOOL HERE


make_pie = RichToolDescription(
    description="Generate a pie chart from labels and values.",
    use_when="Use when you need to create a pie chart visualization from category/value data.",
    side_effects="Generates and returns a PNG pie chart image.",
)


@mcp.tool(description=make_pie.model_dump_json())
async def make_pie_chart(
    labels: Annotated[
        list[str], Field(description="List of category labels for the pie chart")
    ],
    values: Annotated[
        list[float], Field(description="List of numeric values for each category")
    ],
    title: Annotated[
        Optional[str], Field(description="Optional title for the pie chart")
    ] = "",
    width: Annotated[int, Field(description="Image width in pixels")] = 1000,
    height: Annotated[int, Field(description="Image height in pixels")] = 600,
) -> list[ImageContent]:
    import io
    import base64
    from PIL import Image, ImageDraw, ImageFont
    import math

    if len(labels) != len(values):
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message="Labels and values must have the same length.",
            )
        )

    modern_colors = [
        "#4F46E5",
        "#06B6D4",
        "#10B981",
        "#F59E0B",
        "#EF4444",
        "#8B5CF6",
        "#EC4899",
        "#84CC16",
        "#F97316",
        "#6366F1",
        "#14B8A6",
        "#A855F7",
    ]

    colors = modern_colors * ((len(labels) // len(modern_colors)) + 1)
    colors = colors[: len(labels)]

    img = Image.new("RGB", (width, height), "#FAFAFA")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("gsans.ttf", 24)
        legend_font = ImageFont.truetype("gsans.ttf", 20)
        value_font = ImageFont.truetype("gsans.ttf", 18)
    except:
        title_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()
        value_font = ImageFont.load_default()

    chart_size = min(height - 100, width * 0.6) - 80
    chart_x = 80
    chart_y = (height - chart_size) // 2

    if title:
        chart_y += 20

    shadow_offset = 4
    shadow_color = "#E5E7EB"
    draw.ellipse(
        [
            chart_x + shadow_offset,
            chart_y + shadow_offset,
            chart_x + chart_size + shadow_offset,
            chart_y + chart_size + shadow_offset,
        ],
        fill=shadow_color,
    )

    total = sum(values)
    start_angle = -90

    slice_data = []
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        percentage = (value / total) * 100
        angle = (value / total) * 360
        end_angle = start_angle + angle

        draw.pieslice(
            [chart_x, chart_y, chart_x + chart_size, chart_y + chart_size],
            start=start_angle,
            end=end_angle,
            fill=color,
            outline="#FFFFFF",
            width=3,
        )

        mid_angle = math.radians((start_angle + end_angle) / 2)
        slice_data.append(
            {
                "label": label,
                "value": value,
                "percentage": percentage,
                "color": color,
                "mid_angle": mid_angle,
            }
        )

        start_angle = end_angle

    legend_x = chart_x + chart_size + 60
    legend_y = chart_y + 20
    legend_spacing = 40

    draw.text((legend_x, legend_y - 25), "Legend", fill="#374151", font=legend_font)

    for i, slice_info in enumerate(slice_data):

        indicator_size = 18
        draw.rectangle(
            [
                legend_x,
                legend_y + i * legend_spacing,
                legend_x + indicator_size,
                legend_y + i * legend_spacing + indicator_size,
            ],
            fill=slice_info["color"],
            outline="#E5E7EB",
            width=1,
        )

        label_text = f"{slice_info['label']}"
        percentage_text = f"{slice_info['percentage']:.1f}%"
        value_text = f"({slice_info['value']:.0f})"

        draw.text(
            (legend_x + 30, legend_y + i * legend_spacing),
            label_text,
            fill="#1F2937",
            font=legend_font,
        )

        draw.text(
            (legend_x + 30, legend_y + i * legend_spacing + 18),
            f"{percentage_text} {value_text}",
            fill="#6B7280",
            font=value_font,
        )

    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = bbox[2] - bbox[0]
        title_x = (width - title_width) // 2
        title_y = 20

        draw.text((title_x + 1, title_y + 1), title, fill="#D1D5DB", font=title_font)
        draw.text((title_x, title_y), title, fill="#111827", font=title_font)

    draw.rectangle([0, 0, width - 1, height - 1], outline="#E5E7EB", width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return [ImageContent(type="image", mimeType="image/png", data=img_b64)]


# make scatter plot tool here


make_scatter_plot = RichToolDescription(
    description="Generate a labeled scatter plot image from x and y coordinate data.",
    use_when="Use when you want a visual (PNG) representing the relationship between two numeric variables.",
    side_effects="No persistent side effects; returns an image to the caller.",
)


@mcp.tool(description=make_scatter_plot.model_dump_json())
async def make_scatter_plot(
    x_values: Annotated[List[float], Field(description="List of x-coordinate values.")],
    y_values: Annotated[
        List[float],
        Field(description="List of y-coordinate values (same length as x_values)."),
    ],
    labels: Annotated[
        Optional[List[str]], Field(description="Optional list of point labels")
    ] = None,
    categories: Annotated[
        Optional[List[str]],
        Field(description="Optional list of categories for color coding"),
    ] = None,
    colors: Annotated[
        Optional[List[str]],
        Field(description="Optional list of colors for each point (hex codes)"),
    ] = None,
    width: Annotated[int, Field(description="Image width in px (optional)")] = 900,
    height: Annotated[int, Field(description="Image height in px (optional)")] = 600,
    bg_color: Annotated[
        Optional[str], Field(description="Background color, e.g. 'white'")
    ] = "#fafafa",
    point_color: Annotated[
        Optional[str], Field(description="Default point color")
    ] = "#4F46E5",
    point_size: Annotated[int, Field(description="Point radius in pixels")] = 6,
    title: Annotated[Optional[str], Field(description="Optional chart title")] = None,
    x_label: Annotated[
        Optional[str], Field(description="Optional x-axis label")
    ] = None,
    y_label: Annotated[
        Optional[str], Field(description="Optional y-axis label")
    ] = None,
) -> list[TextContent | ImageContent]:
    modern_colors = [
        "#4F46E5",
        "#06B6D4",
        "#10B981",
        "#F59E0B",
        "#EF4444",
        "#8B5CF6",
        "#EC4899",
        "#84CC16",
        "#F97316",
        "#6366F1",
        "#14B8A6",
        "#A855F7",
        "#DC2626",
        "#059669",
        "#0284C7",
        "#7C3AED",
        "#BE185D",
        "#CA8A04",
        "#E11D48",
        "#0891B2",
    ]
    try:
        if x_values is None or y_values is None:
            raise ValueError("Both 'x_values' and 'y_values' must be provided.")

        x_vals = list(x_values)
        y_vals = list(y_values)

        if len(x_vals) != len(y_vals):
            raise ValueError(
                "Length mismatch: 'x_values' and 'y_values' must have the same length."
            )
        if len(x_vals) == 0:
            raise ValueError("Empty lists provided. Provide at least one x/y pair.")

        MAX_POINTS = 2000
        if len(x_vals) > MAX_POINTS:
            raise ValueError(
                f"Too many points ({len(x_vals)}). Max supported is {MAX_POINTS}."
            )

        numeric_x = []
        numeric_y = []
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            try:
                numeric_x.append(float(x))
                numeric_y.append(float(y))
            except Exception:
                raise ValueError(
                    f"Non-numeric data value found at index {i}: x={x}, y={y}"
                )

        if labels is not None and len(labels) != len(x_vals):
            raise ValueError(
                "Length mismatch: 'labels' must have the same length as data points."
            )
        if categories is not None and len(categories) != len(x_vals):
            raise ValueError(
                "Length mismatch: 'categories' must have the same length as data points."
            )
        if colors is not None and len(colors) != len(x_vals):
            raise ValueError(
                "Length mismatch: 'colors' must have the same length as data points."
            )

        def _get_darker_variant(color: str) -> str:
            if color.startswith("#") and len(color) == 7:
                try:
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)

                    r = max(0, int(r * 0.7))
                    g = max(0, int(g * 0.7))
                    b = max(0, int(b * 0.7))

                    return f"#{r:02x}{g:02x}{b:02x}"
                except:
                    pass
            return "#2d2d2d"

        def _measure_text(
            draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
        ):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return w, h

        def _format_number(value: float) -> str:
            if abs(value) >= 1000000:
                return f"{value/1000000:.1f}M"
            elif abs(value) >= 1000:
                return f"{value/1000:.1f}K"
            elif abs(value) >= 100:
                return f"{value:.0f}"
            elif abs(value) >= 1:
                return f"{value:.1f}"
            else:
                return f"{value:.2f}"

        def _draw_rotated_text(
            img: Image.Image,
            text: str,
            x: int,
            y: int,
            font: ImageFont.ImageFont,
            fill: str,
            angle: int = 90,
        ):
            txt_img = Image.new("RGBA", (200, 50), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((0, 0), text, font=font, fill=fill)

            rotated = txt_img.rotate(angle, expand=1)

            paste_x = x - rotated.width // 2
            paste_y = y - rotated.height // 2

            img.paste(rotated, (paste_x, paste_y), rotated)

        def _create_scatter_plot_image(
            x_data: List[float], y_data: List[float]
        ) -> bytes:
            pad_x = 100
            pad_y = 80
            title_space = 60 if title else 30
            x_label_space = 40 if x_label else 20
            y_label_space = 60 if y_label else 40

            img_w = max(400, width)
            img_h = max(300, height)

            chart_w = img_w - pad_x - y_label_space
            chart_h = img_h - pad_y - title_space - x_label_space

            img = Image.new("RGB", (img_w, img_h), color=bg_color)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("gsans.ttf", 20)
                title_font = ImageFont.truetype("gsans.ttf", 24)
                label_font = ImageFont.truetype("gsans.ttf", 20)
            except:
                try:
                    font = ImageFont.load_default()
                    title_font = ImageFont.load_default()
                    label_font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                    title_font = font
                    label_font = font

            min_x, max_x = min(x_data), max(x_data)
            min_y, max_y = min(y_data), max(y_data)

            x_range = max_x - min_x
            y_range = max_y - min_y

            if x_range == 0:
                x_range = 1
                min_x -= 0.5
                max_x += 0.5
            else:
                padding_x = x_range * 0.05
                min_x -= padding_x
                max_x += padding_x
                x_range = max_x - min_x

            if y_range == 0:
                y_range = 1
                min_y -= 0.5
                max_y += 0.5
            else:
                padding_y = y_range * 0.05
                min_y -= padding_y
                max_y += padding_y
                y_range = max_y - min_y

            chart_x = y_label_space
            chart_y = title_space
            chart_right = chart_x + chart_w
            chart_bottom = chart_y + chart_h

            grid_color = "#e5e7eb"
            axis_color = "#9ca3af"

            x_ticks = 6
            for i in range(x_ticks + 1):
                x_val = min_x + (max_x - min_x) * (i / x_ticks)
                x_pos = chart_x + (x_val - min_x) / x_range * chart_w

                if 0 < i < x_ticks:
                    draw.line(
                        [(x_pos, chart_y), (x_pos, chart_bottom)],
                        fill=grid_color,
                        width=1,
                    )

                label_text = _format_number(x_val)
                lbl_w, lbl_h = _measure_text(draw, label_text, font)
                draw.text(
                    (x_pos - lbl_w / 2, chart_bottom + 8),
                    label_text,
                    font=font,
                    fill="#6b7280",
                )

            y_ticks = 6
            for i in range(y_ticks + 1):
                y_val = min_y + (max_y - min_y) * (i / y_ticks)
                y_pos = chart_bottom - (y_val - min_y) / y_range * chart_h

                if 0 < i < y_ticks:
                    draw.line(
                        [(chart_x, y_pos), (chart_right, y_pos)],
                        fill=grid_color,
                        width=1,
                    )

                label_text = _format_number(y_val)
                lbl_w, lbl_h = _measure_text(draw, label_text, font)
                draw.text(
                    (chart_x - lbl_w - 8, y_pos - lbl_h / 2),
                    label_text,
                    font=font,
                    fill="#6b7280",
                )

            draw.line(
                [(chart_x, chart_y), (chart_x, chart_bottom)], fill=axis_color, width=2
            )
            draw.line(
                [(chart_x, chart_bottom), (chart_right, chart_bottom)],
                fill=axis_color,
                width=2,
            )

            if categories:
                unique_categories = list(set(categories))
                category_colors = {
                    cat: modern_colors[i % len(modern_colors)]
                    for i, cat in enumerate(unique_categories)
                }
            else:
                category_colors = {}

            for i, (x_val, y_val) in enumerate(zip(x_data, y_data)):
                x_pos = chart_x + (x_val - min_x) / x_range * chart_w
                y_pos = chart_bottom - (y_val - min_y) / y_range * chart_h

                if colors and i < len(colors):
                    color = colors[i]
                elif categories:
                    color = category_colors[categories[i]]
                else:
                    color = (
                        modern_colors[i % len(modern_colors)]
                        if i < len(modern_colors)
                        else point_color
                    )

                border_color = _get_darker_variant(color)

                draw.ellipse(
                    [
                        x_pos - point_size,
                        y_pos - point_size,
                        x_pos + point_size,
                        y_pos + point_size,
                    ],
                    fill=color,
                    outline=border_color,
                    width=2,
                )

                highlight_size = max(1, point_size // 3)
                draw.ellipse(
                    [
                        x_pos - highlight_size,
                        y_pos - highlight_size,
                        x_pos + highlight_size,
                        y_pos + highlight_size,
                    ],
                    fill="#ffffff",
                    outline=None,
                )

            if categories and len(set(categories)) > 1:
                legend_x = chart_right - 150
                legend_y = chart_y + 20
                legend_spacing = 25

                for i, (cat, color) in enumerate(category_colors.items()):
                    y_pos = legend_y + i * legend_spacing

                    draw.ellipse(
                        [legend_x, y_pos, legend_x + 12, y_pos + 12],
                        fill=color,
                        outline=_get_darker_variant(color),
                        width=1,
                    )

                    draw.text(
                        (legend_x + 20, y_pos - 2), str(cat), font=font, fill="#374151"
                    )

            if title:
                t_w, t_h = _measure_text(draw, title, title_font)
                draw.text(
                    ((img_w - t_w) / 2, 15), title, font=title_font, fill="#111827"
                )

            if x_label:
                x_w, x_h = _measure_text(draw, x_label, label_font)
                draw.text(
                    ((img_w - x_w) / 2, img_h - 25),
                    x_label,
                    font=label_font,
                    fill="#374151",
                )

            if y_label:
                _draw_rotated_text(
                    img, y_label, 30, img_h // 2, label_font, "#374151", 90
                )

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        png_bytes = await asyncio.to_thread(
            _create_scatter_plot_image, numeric_x, numeric_y
        )
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return [ImageContent(type="image", mimeType="image/png", data=b64)]

    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# MEDICINE DATA TOOL HERE


async def _get_medicine_page(url: str) -> BeautifulSoup | None:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
    except httpx.RequestError:
        return None


def _safe_get_text(
    soup: BeautifulSoup, selector: str, default: str = "Not found"
) -> str:
    tag = soup.select_one(selector)
    return tag.get_text(strip=True) if tag else default


async def _find_medicine_on_1mg(
    medicine_name: str,
) -> Tuple[str, str] | Tuple[None, None]:
    search_url = (
        f"https://www.1mg.com/search/all?name={medicine_name.replace(' ', '%20')}"
    )
    soup = await _get_medicine_page(search_url)
    if not soup:
        return None, None
    link_selector = (
        "div[class*='style__horizontal-card'] > a, a[class*='style__product-link']"
    )
    product_link_tag = soup.select_one(link_selector)
    if not product_link_tag or not product_link_tag.get("href"):
        return None, None
    product_url = "https://www.1mg.com" + product_link_tag["href"]
    name_selector = "div[class*='style__pro-title'], span[class*='style__pro-title']"
    official_name = _safe_get_text(product_link_tag, name_selector, medicine_name)
    return official_name, product_url


FIND_MEDICINE_DESC = RichToolDescription(
    description="Finds detailed information for a specific medicine, including its official name, uses, side effects, price, and substitutes.",
    use_when="When a user asks for details about a specific medicine by name.",
    side_effects="Fetches live data by scraping the Tata 1mg website.",
)


@mcp.tool(description=FIND_MEDICINE_DESC.model_dump_json())
async def find_medicine_details(
    medicine_name: Annotated[str, Field(description="The name of the medicine.")],
) -> str:
    if not medicine_name.strip():
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message="Medicine name cannot be empty.")
        )
    official_name, product_url = await _find_medicine_on_1mg(medicine_name)
    if not product_url:
        return f"Sorry, I could not find any medicine matching '{medicine_name}'. Please check the spelling."

    p_soup = await _get_medicine_page(product_url)
    if not p_soup:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Network error while scraping product page for {official_name}.",
            )
        )

    details = {
        "official_name": official_name,
        "product_url": product_url,
        "manufacturer": _safe_get_text(p_soup, "a[href*='/marketer/']"),
        "estimated_price": _safe_get_text(
            p_soup, "span[class*='PriceBoxPlanOption__offer-price']"
        ),
        "primary_use": _safe_get_text(
            p_soup, "div#overview div[class*='DrugOverview__content']"
        ),
        "side_effects": _safe_get_text(
            p_soup, "div#side_effects div[class*='DrugOverview__content']"
        ),
        "substitutes": [],
    }
    substitutes_list = p_soup.select("div[class*='SubstituteItem__item']")
    for sub_item in substitutes_list:
        sub_name = _safe_get_text(sub_item, "div[class*='SubstituteItem__name'] a")
        sub_price = _safe_get_text(sub_item, "div[class*='SubstituteItem__unit-price']")
        if sub_name != "Not found":
            details["substitutes"].append({"name": sub_name, "price": sub_price})

    output_yaml = yaml.dump(details, sort_keys=False, allow_unicode=True)
    prompt_for_ai = (
        "\n---\n"
        "Present this information clearly to the user, using the `official_name`. ALWAYS include the `product_url`. "
        "ALWAYS mention the substitutes as a cost-saving option."
    )
    return output_yaml + prompt_for_ai


async def main():
    print("Starting minimal MCP server on http://0.0.0.0:8085")
    try:
        await mcp.run_async("streamable-http", host="0.0.0.0", port=8085)
    finally:
        await asyncio.to_thread(save_data)
        print("Shutdown: data saved.")


if __name__ == "__main__":
    init_data()
    asyncio.run(main())
