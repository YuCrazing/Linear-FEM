import numpy as np

# This function parses the output of TetGen and returns the nodes, faces, and elements. 
# All returned indices are 0-based.
def parse_tetgen_output(model_path_without_extension):
    node_file = model_path_without_extension + '.node'

    with open(node_file, 'r') as f:
        lines = f.readlines()

    # Skip the first line (header)
    vn = int(lines[0].strip().split()[0])
    lines = lines[1:-1]


    nodes = []
    for line in lines:
        parts = line.strip().split()
        node_number = int(parts[0])
        # print(parts)
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        # nodes.append((node_number, x, y, z))
        nodes.append((x, y, z))

    # Now 'nodes' contains a list of tuples (node_number, x, y, z)
    assert len(nodes) == vn


    face_file = model_path_without_extension + '.face'

    with open(face_file, 'r') as f:
        lines = f.readlines()

    # Skip the first line (header)
    fn = int(lines[0].strip().split()[0])
    lines = lines[1:-1]

    faces = []
    for line in lines:
        parts = line.strip().split()
        face_number = int(parts[0])
        # In this example, we assume that TetGen generates triangle faces with 3 nodes per face.
        node_indices = list(map(int, parts[1:-1]))
        # faces.append((face_number, node_indices))
        faces.append((node_indices))

    # Now 'faces' contains a list of tuples (face_number, node_indices)
    assert len(faces) == fn


    ele_file = model_path_without_extension + '.ele'

    with open(ele_file, 'r') as f:
        lines = f.readlines()

    # Skip the first line (header)
    en = int(lines[0].strip().split()[0])
    lines = lines[1:-1]

    elements = []
    for line in lines:
        parts = line.strip().split()
        element_number = int(parts[0])
        # In this example, we assume that TetGen generates tetrahedral elements with 4 nodes per element.
        # node_indices = list(map(int, parts[1:]))
        node_indices = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
        # elements.append((element_number, node_indices))
        elements.append((node_indices))

    # Now 'elements' contains a list of tuples (element_number, node_indices)
    assert len(elements) == en

    print("tetgen parser: vn:", len(nodes), "fn:", len(faces), "en:", len(elements))
    # print(nodes, faces, elements)

    # Convert from 1-based indexing to 0-based indexing 
    return np.array(nodes), np.array(faces)-1, np.array(elements)-1

if __name__ == '__main__':
    parse_tetgen_output('data/example.1')