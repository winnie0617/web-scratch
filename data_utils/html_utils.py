"""
Reference: https://github.com/nat/natbot/blob/main/natbot.py
"""

salient_attributes = ["type", "placeholder", "aria_label", "title", "alt", "class"]
black_listed_elements = set(["html", "head", "title", "meta", "iframe", "body", "script", "style", "path", "svg", "br", "::marker",])

def add_to_hash_tree(hash_tree, tag, node):
    
    node_id = node.attrib.get("backend_node_id", "-1")
    
    parent = node.getparent()
    if parent is None:
        hash_tree[str(node_id)] = (False, None)
        return (False, None)
    
    parent_id_str = parent.get("backend_node_id", "-1")
    if parent_id_str == "-1":
        hash_tree[str(node_id)] = (False, None)
        return (False, None)
    
    if not parent_id_str in hash_tree:
        add_to_hash_tree(hash_tree, tag, parent)

    is_parent_desc_anchor, anchor_id = hash_tree[parent_id_str]

    # even if the anchor is nested in another anchor, we set the "root" for all descendants to be ::Self
    
    if node.tag == tag: # account for div role="button"
        value = (True, node_id)
    elif (
        is_parent_desc_anchor
    ):  # reuse the parent's anchor_id (which could be much higher in the tree)
        value = (True, anchor_id)
    else:
        value = (
            False,
            None,
        )  # not a descendant of an anchor, most likely it will become text, an interactive element or discarded

    hash_tree[str(node_id)] = value

    return value

def convert_name(node_name, has_click_handler):
    if node_name == "a":
        return "link"
    if node_name == "input":
        return "input"
    if node_name == "img":
        return "img"
    if (
        node_name == "button" or has_click_handler
    ):  # found pages that needed this quirk
        return "button"
    else:
        return "text"
    
def prune_dom_tree(dom_tree):
    # Initialize caches
    child_nodes = {}
    anchor_ancestry = {"-1": (False, None)}
    button_ancestry = {"-1": (False, None)}

    elements = []

    # iterate over all nodes in dom_tree
    for node in dom_tree.iter():
        
        if "backend_node_id" not in node.attrib:
            continue
        
        # to deal with div role="button" case
        if  node.attrib.get("role") == "button":
            node.tag = "button"
        # this means ancestor is a button?
        is_ancestor_of_anchor, anchor_id = add_to_hash_tree(anchor_ancestry, "a", node)
        is_ancestor_of_button, button_id = add_to_hash_tree(button_ancestry, "button", node)
        
        # Skip blacklisted elements
        if node.tag in black_listed_elements:
            continue

        # add attributes if exist
        element_attributes = {}
        for key in node.attrib:
            if key in salient_attributes:
                element_attributes[key] = node.attrib[key]
                        
        # Determine if node is an ancestor of anchor or button
        ancestor_exception = is_ancestor_of_anchor or is_ancestor_of_button
        # get key of ancestor node
        ancestor_node_key = (
            None
            if not ancestor_exception
            else anchor_id
            if is_ancestor_of_anchor
            else button_id
        )
        # access ancestor of ancestor if exist, else initialize with value equal empty
        # list is for storing attributes of the child nodes
        ancestor_node = (
            None
            if not ancestor_exception
            else child_nodes.setdefault(ancestor_node_key, [])
        )
        
            
        meta_data = []

        if node.tag == "text" and ancestor_exception:
            text = node.text.strip()
            if text == "|" or text == "•":
                continue
            ancestor_node.append({
                "type": "type", "value": text
            })
        else:
            if (
                node.tag == "input" and element_attributes.get("type") == "submit"
            ) or node.tag == "button":
                node.tag = "button"
                element_attributes.pop(
                    "type", None
                )  # prevent [button ... (button)..]
            
            for key in element_attributes:
                if ancestor_exception:
                    ancestor_node.append({
                        "type": "attribute",
                        "key":  key,
                        "value": element_attributes[key]
                    })
                else:
                    meta_data.append(element_attributes[key])

        element_node_value = None

        # TODO

        element_node_value = node.text

        if element_node_value == "|": #commonly used as a seperator, does not add much context - lets save ourselves some token space
            continue
    # if (
    #     node_name == "input"
    #     and index in input_value_index
    #     and element_node_value is None
    # ):
    #     node_input_text_index = input_value_index.index(index)
    #     text_index = input_value_values[node_input_text_index]
    #     if node_input_text_index >= 0 and text_index >= 0:
    #         element_node_value = strings[text_index]
        

        # remove redudant elements
        if ancestor_exception and (node.tag != "a" and node.tag != "button"):
            continue

        # is_clickable = node.attrib.get("role") == "button" or node.attrib.get("role") == "link"
        elements.append(
            {
                # "node_index": str(index),
                "backend_node_id": node.attrib["backend_node_id"],
                "node_name": node.tag,
                "node_value": element_node_value,
                "node_meta": meta_data,
                # "is_clickable": is_clickable,
                # "origin_x": int(x),
                # "origin_y": int(y),
                # "center_x": int(x + (width / 2)),
                # "center_y": int(y + (height / 2)),
            }
        )
        elements_of_interest= []
        # Use original backend_node_id instead of id_counter for easier matching back to the original element
        # id_counter 			= 0

    for element in elements:
        back_node_id = element.get("backend_node_id")
        node_name = element.get("node_name")
        node_value = element.get("node_value")
        # is_clickable = element.get("is_clickable")
        # origin_x = element.get("origin_x")
        # origin_y = element.get("origin_y")
        # center_x = element.get("center_x")
        # center_y = element.get("center_y")
        meta_data = element.get("node_meta")

        inner_text = f"{node_value} " if node_value else ""
        meta = ""
            
        if back_node_id in child_nodes:
            for child in child_nodes.get(back_node_id):
                entry_type = child.get('type')
                entry_value= child.get('value')
                if entry_type == "attribute":
                    entry_key = child.get('key')
                    meta_data.append(f'{entry_key}="{entry_value}"')
                else:
                    inner_text += f"{entry_value} "

        if meta_data:
            meta_string = " ".join(meta_data)
            meta = f" {meta_string}"

        if inner_text != "":
            inner_text = f"{inner_text.strip()}"
        
        else:
            # use ariel label if available

            if meta_data and "aria-label" in meta_data:
                inner_text = meta_data["aria-label"].strip()

        converted_node_name = convert_name(node_name, False)

            
        # skip elements that are not of interest
        # not very elegant, more like a placeholder
        if (
            (converted_node_name != "button" or meta == "") # empty button (?)
            and converted_node_name != "link"
            and converted_node_name != "input"
            and converted_node_name != "img"
            and converted_node_name != "textarea"
        ) and inner_text.strip() == "":
            continue
        
        # page_element_buffer[id_counter] = element

        # Format elements into HTML-like tags
        if inner_text != "": 
            elements_of_interest.append(
                f"""<{converted_node_name} id={back_node_id}{meta}>{inner_text}</{converted_node_name}>"""
            )
        else:
            elements_of_interest.append(
                f"""<{converted_node_name} id={back_node_id}{meta}/>"""
            )
        # id_counter += 1
    
    return elements_of_interest