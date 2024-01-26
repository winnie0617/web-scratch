from playwright.sync_api import sync_playwright
import lxml
from lxml import etree
from lxml import html
from lxml.html.clean import Cleaner

import re

def build_dom_tree(document, documents, str_mapping):
    dom_nodes = document["nodes"]
    layout_nodes = document["layout"]
    array_parent_index = dom_nodes["parentIndex"]
    array_node_type = dom_nodes["nodeType"]
    array_node_name = dom_nodes["nodeName"]
    array_node_value = dom_nodes["nodeValue"]
    array_backend_node_id = dom_nodes["backendNodeId"]
    array_attributes = dom_nodes["attributes"]
    dict_text_value = {idx: value for idx, value in zip(dom_nodes["textValue"]["index"], dom_nodes["textValue"]["value"])}
    dict_input_value = {idx: value for idx, value in zip(dom_nodes["inputValue"]["index"], dom_nodes["inputValue"]["value"])}
    set_input_checked = set(dom_nodes["inputChecked"]["index"])
    set_option_selected = set(dom_nodes["optionSelected"]["index"])
    dict_content_document_index = {idx: value for idx, value in zip(dom_nodes["contentDocumentIndex"]["index"], dom_nodes["contentDocumentIndex"]["value"])}
    dict_pseudo_type = {idx: value for idx, value in zip(dom_nodes["pseudoType"]["index"], dom_nodes["pseudoType"]["value"])}
    set_is_clickable = set(dom_nodes["isClickable"]["index"])
    
    dict_layout = {node_idx: bound for node_idx, bound in zip(layout_nodes["nodeIndex"], layout_nodes["bounds"])}
    
    def get_str(str_idx):
        if str_idx == -1:
            return ""
        else:
            return str_mapping[str_idx]
    node_elements = []
    for node_idx in range(len(array_node_name)):
        node_name = get_str(array_node_name[node_idx]).lower()
        if node_name == "#document":
            node_name = "ROOT_DCOUMENT"
        elif node_name == "#text":
            node_name = "text"
        elif node_name.startswith("::"):
            node_name = node_name.replace("::", "pseudo-")
        elif node_name.startswith("#"):
            node_name = node_name.replace("#", "hash-")
        node_name = re.sub(r'[^\w\s]', '-', node_name)
        node_element = etree.Element(node_name)
        node_value = get_str(array_node_value[node_idx])
        node_element.text = node_value
        node_element.set("backend_node_id", str(array_backend_node_id[node_idx]))
        node_element.set("bounding_box_rect", ",".join([str(x) for x in dict_layout.get(node_idx, [-1, -1, -1, -1])]))
        for attr_idx in range(0, len(array_attributes[node_idx]), 2):
            attr_name = re.sub(r'[^\w]', '_', get_str(array_attributes[node_idx][attr_idx]))
            if attr_name[0].isdigit():
                attr_name = "_"+attr_name
            attr_value = get_str(array_attributes[node_idx][attr_idx+1])
            node_element.set(attr_name, attr_value)
        if node_idx in dict_text_value:
            node_element.set("text_value", get_str(dict_text_value[node_idx]))
        if node_idx in dict_input_value:
            node_element.set("input_value", get_str(dict_input_value[node_idx]))
        if node_idx in set_input_checked:
            node_element.set("input_checked", "true")
        if node_idx in set_option_selected:
            node_element.set("option_selected", "true")
        if node_idx in dict_pseudo_type:
            node_element.set("pseudo_type", get_str(dict_pseudo_type[node_idx]))
        if node_idx in set_is_clickable:
            node_element.set("is_clickable", "true")
            
        if node_idx in dict_content_document_index:
            iframe_dom_tree = build_dom_tree(documents[dict_content_document_index[node_idx]], documents, str_mapping)
            node_element.append(iframe_dom_tree)
        parent_node_idx = array_parent_index[node_idx]
        if array_parent_index[node_idx] != -1:
            node_elements[parent_node_idx].append(node_element)
        node_elements.append(node_element)

    html_root = [e for e in node_elements if len(e)!=0 and e.tag=="html"]
    if html_root:
        return html_root[0]
    else:
        return node_elements[0]
    
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