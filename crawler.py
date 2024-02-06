from playwright.sync_api import sync_playwright
import lxml
from lxml import etree
from lxml import html
from lxml.html.clean import Cleaner

import re

from data_utils.dom_utils import build_dom_tree

VALID_ATTRIBS = {
    'alt',
    'aria_description',
    'aria_label',
    'aria_role',
    'backend_node_id',
    'bounding_box_rect',
    'class',
    'data_pw_testid_buckeye',
    'id',
    'input_checked',
    'input_value',
    'is_clickable',
    'label',
    'name',
    'option_selected',
    'placeholder',
    'pseudo_type',
    'role',
    'text_value',
    'title',
    'type',
    'value'
}

cleaner = Cleaner(
    scripts=True,  # Removes any <script> tags.
    javascript=True,  # Removes any Javascript, like an onclick attribute. Also removes stylesheets as they could contain Javascript.
    comments=True,  # Removes any comments.
    style=True,  # Removes any style tags.
    inline_style=True,  # Removes any style attributes. Defaults to the value of the style option.
    links=True,  # Removes any <link> tags
    meta=True,  # Removes any <meta> tags
    page_structure=False,  # Structural parts of a page: <head>, <html>, <title>.
    processing_instructions=True,  # Removes any processing instructions.
    embedded=False,  # Removes any embedded objects (flash, iframes)
    frames=False,  # Removes any frame-related tags
    forms=False,  # Removes any form tags
    annoying_tags=True,  # Tags that aren't wrong, but are annoying. <blink> and <marquee>
    remove_tags=None,  # A list of tags to remove. Only the tags will be removed, their content will get pulled up into the parent tag.
    remove_unknown_tags=False,
    safe_attrs_only=True,  # If true, only include 'safe' attributes (specifically the list from the feedparser HTML sanitisation web site).
    safe_attrs=list(
        VALID_ATTRIBS
    ),  # A set of attribute names to override the default list of attributes considered 'safe' (when safe_attrs_only=True).
)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https:seatgeek.com/")
    page.screenshot(path="example.png")
    
    client = page.context.new_cdp_session(page)
    tree = client.send(
        "DOMSnapshot.captureSnapshot",
        {"computedStyles": []},
    )
    content = client.send("Page.captureSnapshot")
    root = build_dom_tree(tree["documents"][0], tree["documents"], tree["strings"])
    root = lxml.html.fromstring(lxml.html.tostring(root, encoding="unicode"))
    cleaner(root)
    html = lxml.html.tostring(root, pretty_print=True, encoding="unicode")
    # save html to file
    with open("example.html", "w") as f:
        f.write(html)
    
    browser.close()