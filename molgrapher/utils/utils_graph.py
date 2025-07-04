import copy
import json
import math
import os
import time
from collections import defaultdict
from pprint import pprint

import networkx as nx
import torch
import torch_geometric
from matplotlib import colors as mcolors
from mol_depict.utils.utils_drawing import draw_molecule_rdkit
from rdkit import Chem
from torch_geometric.data import Data

from molgrapher.utils.utils_postprocessing import GraphPostprocessor


class MolecularGraph:
    def __init__(self, config_dataset_graph, filename_logging=""):
        """
        Generic graph structure to ease conversions.
            - Atoms: Molecule's atoms.
            - Bonds: Molecule's bonds
            - Node: Graph's nodes.
            - Edge: Graph's edges.
            - Data: Pytorch Geometric graph structure
        """
        self.atoms = []
        self.bonds = []
        self.nodes = []
        self.edges = []
        self.data_nodes_only = None
        self.data = None
        self.bond_size = 0
        self.config_dataset_graph = config_dataset_graph
        self.symbols_classes_atoms = json.load(
            open(
                os.path.dirname(__file__)
                + f"/../../data/vocabularies/vocabulary_atoms_{config_dataset_graph['nb_atoms_classes']}.json"
            )
        )
        self.types_classes_bonds = json.load(
            open(
                os.path.dirname(__file__)
                + f"/../../data/vocabularies/vocabulary_bonds_{config_dataset_graph['nb_bonds_classes']}.json"
            )
        )
        self.atoms_classes_symbols = {
            v: k for k, v in self.symbols_classes_atoms.items()
        }
        self.bonds_classes_types = {v: k for k, v in self.types_classes_bonds.items()}
        self.filename_logging = filename_logging

    def from_cede(self, annotation, keypoints):
        atom_index = 0
        bond_index = 0
        for i, a in annotation.iterrows():
            if isinstance(a["element"], str) or (not math.isnan(a["element"])):
                # Atom
                atom_symbol = a["element"]
                if atom_symbol == "pseudoatom":
                    atom_symbol = "R"
                if a["charge"] != 0:
                    atom_symbol = atom_symbol + "," + str(int(a["charge"]))
                if atom_symbol not in self.symbols_classes_atoms:
                    print(f"Atom type: {atom_symbol} rejected")
                    atom_symbol = "None"

                self.atoms.append(
                    {
                        "index": atom_index,
                        "class": self.symbols_classes_atoms[atom_symbol],
                        "position": [
                            keypoints[atom_index][0],
                            keypoints[atom_index][1],
                        ],
                        "type": 1,
                    }
                )
                atom_index += 1

            if isinstance(a["connecting_nodes"], list) or (
                not math.isnan(a["connecting_nodes"])
            ):
                b, e = a["connecting_nodes"]
                b_i = (annotation[annotation.id == b].index - annotation.index[0])[0]
                e_i = (annotation[annotation.id == e].index - annotation.index[0])[0]
                begin_position = [keypoints[b_i][0], keypoints[b_i][1]]
                end_position = [keypoints[e_i][0], keypoints[e_i][1]]

                if bond_index == 0:
                    self.bond_size = math.sqrt(
                        (begin_position[0] - end_position[0]) ** 2
                        + (begin_position[1] - end_position[1]) ** 2
                    )

                bond_type = a["smarts"]
                if (
                    bond_type == "-"
                    or bond_type == "up"
                    or bond_type == "down"
                    or bond_type == "/"
                    or bond_type == "\\"
                ):
                    bond_type = "SINGLE"
                elif bond_type == "=" or bond_type == "either":
                    bond_type = "DOUBLE"
                elif bond_type == "#":
                    bond_type = "TRIPLE"
                if bond_type not in self.types_classes_bonds:
                    print(f"Bond type: {bond_type} rejected")
                    bond_type = "None"

                self.bonds.append(
                    {
                        "index": [b_i, e_i],
                        "class": self.types_classes_bonds[bond_type],
                        "position": [
                            int((begin_position[0] + end_position[0]) / 2),
                            int((begin_position[1] + end_position[1]) / 2),
                        ],
                        "type": 0,
                    }
                )
                bond_index += 1
        return self

    def from_rdkit_molecule(self, molecule, keypoints):
        if molecule is None:
            print("The Molfile can't be read")
            return

        # Remove aromatic bonds (Kekulize)
        # molecule = rdMolDraw2D.PrepareMolForDrawing(molecule, addChiralHs=False)

        # Set wedge bonds
        # rdDepictor.Compute2DCoords(molecule)
        Chem.WedgeMolBonds(
            molecule, molecule.GetConformers()[0]
        )  # Maybe it is the problem (?)

        # Read atoms
        for atom_index, atom in enumerate(molecule.GetAtoms()):
            if atom.HasProp("_displayLabel"):
                # Probably unecessary
                atom.SetAtomicNum(0)

            keypoint = keypoints[atom_index]
            atom_symbol = atom.GetSymbol()
            if atom_symbol == "*":
                # Probably uncessary
                atom_symbol = "R"
            elif atom.GetFormalCharge():
                atom_symbol = atom.GetSymbol() + "," + str(atom.GetFormalCharge())
            if atom_symbol not in self.symbols_classes_atoms:
                atom_symbol = "None"

            self.atoms.append(
                {
                    "index": atom_index,
                    "class": self.symbols_classes_atoms[atom_symbol],
                    "position": [keypoint[0], keypoint[1]],
                    "type": 1,
                }
            )

        # Read bonds
        for bond_index, bond in enumerate(molecule.GetBonds()):
            b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            begin_position = [keypoints[b][0], keypoints[b][1]]
            end_position = [keypoints[e][0], keypoints[e][1]]

            if bond_index == 0:
                self.bond_size = math.sqrt(
                    (begin_position[0] - end_position[0]) ** 2
                    + (begin_position[1] - end_position[1]) ** 2
                )

            bond_type = str(bond.GetBondType())

            if bond_type not in self.types_classes_bonds:
                bond_type = "None"

            if str(bond.GetBondDir()) == "BEGINWEDGE":
                bond_type = "SOLID"

            if str(bond.GetBondDir()) == "BEGINDASH":
                bond_type = "DASHED"

            self.bonds.append(
                {
                    "index": [b, e],
                    "class": self.types_classes_bonds[bond_type],
                    "position": [
                        int((begin_position[0] + end_position[0]) / 2),
                        int((begin_position[1] + end_position[1]) / 2),
                    ],
                    "type": 0,
                }
            )
        return self

    def set_nodes_edges_nodes_only(self):
        # Convert bonds to nodes
        nb_atoms = len(self.atoms)
        self.nodes = self.atoms.copy()
        edges = []
        for index, bond in enumerate(self.bonds):
            bond_node = bond.copy()
            # Create new bond nodes, appended after atom nodes
            bond_node["index"] = index + nb_atoms
            self.nodes.append(bond_node)

            # Create new edges
            b, e = bond["index"]
            # atoms_edge = {"index": [b, e]}
            atom_b_bond_edge = {"index": [b, bond_node["index"]]}
            bond_atom_e_edge = {"index": [bond_node["index"], e]}
            # edges += [atoms_edge, atom_b_bond_edge, bond_atom_e_edge]
            edges += [atom_b_bond_edge, bond_atom_e_edge]
        self.edges = edges

    def set_nodes_edges(self):
        self.nodes = self.atoms.copy()
        self.edges = self.bonds.copy()

    def duplicate_edges(self):
        edges = []
        for edge in self.edges:
            edge_duplicate = edge.copy()
            edge_duplicate["index"] = [edge["index"][1], edge["index"][0]]
            edges += [edge, edge_duplicate]
        self.edges = edges

    def to_torch_nodes_only(self):
        self.duplicate_edges()
        self.data_nodes_only = Data(
            x=torch.tensor([[0] for node in self.nodes]),
            edge_index=torch.tensor([edge["index"] for edge in self.edges])
            .t()
            .to(torch.long)
            .view(2, -1),
            nodes_positions=torch.tensor([node["position"] for node in self.nodes]).to(
                torch.long
            ),
            nodes_classes=torch.tensor([node["class"] for node in self.nodes]),
            nodes_types=torch.tensor([node["type"] for node in self.nodes]),
        )
        return self.data_nodes_only

    def to_torch(self):
        self.duplicate_edges()
        self.data = Data(
            x=torch.tensor([[0] for node in self.nodes]),
            edge_index=torch.tensor([edge["index"] for edge in self.edges])
            .t()
            .to(torch.long)
            .view(2, -1),
            edges_classes=torch.tensor([edge["class"] for edge in self.edges]),
            nodes_positions=torch.tensor([node["position"] for node in self.nodes]).to(
                torch.long
            ),
            nodes_classes=torch.tensor([node["class"] for node in self.nodes]),
            nodes_types=torch.tensor([node["type"] for node in self.nodes]),
        )
        return self.data

    def display_data(self, axis):
        if self.data is None:
            print("Pytorch geometric graph is not defined")
            return

        # Define mappings
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()) * 50
        white_colors = [
            "beige",
            "bisque",
            "blanchedalmond",
            "cornsilk",
            "antiquewhite",
            "ghostwhite",
            "w",
            "whitesmoke",
            "white",
            "snow",
            "seashell",
            "mistyrose",
            "linen",
            "floralwhite",
            "ivory",
            "honeydew",
            "mintcream",
            "azure",
            "aliceblue",
            "lavenderblush",
        ]
        colors = [color for color in colors if color not in white_colors] * 50

        atoms_colors = {k: colors[v] for k, v in self.symbols_classes_atoms.items()}
        bonds_colors = {k: colors[v] for k, v in self.types_classes_bonds.items()}

        # Convert Pytorch graph to Networkx graph
        networkx_graph = torch_geometric.utils.to_networkx(
            self.data,
            to_undirected=True,
            edge_attrs=["edges_classes"],
            node_attrs=["nodes_classes"],
        )

        # Set nodes positions, colors and display labels
        positions = [
            [
                position[0],
                (self.config_dataset_graph["image_size"][1] - 1) - position[1],
            ]
            for position in self.data.nodes_positions.tolist()
        ]
        nodes_colors = []
        nodes_display_labels = {}
        for atom_index, node_class in enumerate(self.data.nodes_classes):
            atom_symbol = self.atoms_classes_symbols[node_class.item()]
            nodes_colors.append(atoms_colors[atom_symbol])
            nodes_display_labels[atom_index] = atom_symbol + ":" + str(atom_index)

        # Set bonds colors and display labels
        edges_colors = []
        edges_display_labels = {}
        for edge in networkx_graph.edges(data=True):
            edge_type = self.bonds_classes_types[edge[2]["edges_classes"]]
            edges_colors.append(bonds_colors[edge_type])
            edges_display_labels[(edge[0], edge[1])] = (
                edge_type[:3] + ":" + str(atom_index)
            )

        # Display graph
        nx.draw(
            networkx_graph,
            positions,
            edge_color=edges_colors,
            width=10,
            linewidths=1,
            node_size=500,
            node_color=nodes_colors,
            alpha=0.3,
            ax=axis,
        )
        nx.draw_networkx_labels(
            networkx_graph,
            positions,
            labels=nodes_display_labels,
            font_color="black",
            font_size=10,
            ax=axis,
        )
        nx.draw_networkx_edge_labels(
            networkx_graph,
            positions,
            edge_labels=edges_display_labels,
            font_color="black",
            font_size=7,
            ax=axis,
            bbox=dict(facecolor="red", alpha=0),
        )

    def display_data_nodes_only(
        self, axis, simple_display=False, supergraph=False, large_molecule_display=True
    ):
        if self.data_nodes_only is None:
            print("Pytorch geometric graph is not defined")
            return

        # Define mappings
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()) * 50
        white_colors = [
            "beige",
            "bisque",
            "blanchedalmond",
            "cornsilk",
            "antiquewhite",
            "ghostwhite",
            "w",
            "whitesmoke",
            "white",
            "snow",
            "seashell",
            "mistyrose",
            "linen",
            "floralwhite",
            "ivory",
            "honeydew",
            "mintcream",
            "azure",
            "aliceblue",
            "lavenderblush",
        ]
        colors = [color for color in colors if color not in white_colors] * 50

        atoms_colors = {k: colors[v] for k, v in self.symbols_classes_atoms.items()}
        bonds_colors = {k: colors[v] for k, v in self.types_classes_bonds.items()}

        # Convert Pytorch graph to Networkx graph
        networkx_graph = torch_geometric.utils.to_networkx(
            self.data_nodes_only, to_undirected=True
        )

        # Set nodes positions, colors and display labels
        positions = [
            [
                position[0],
                ((self.config_dataset_graph["image_size"][1] - 1) - position[1]),
            ]
            for position in self.data_nodes_only.nodes_positions.tolist()
        ]
        atoms_nodes_indices = [
            i for i, type in enumerate(self.data_nodes_only.nodes_types) if type == 1
        ]
        bonds_nodes_indices = [
            i for i, type in enumerate(self.data_nodes_only.nodes_types) if type == 0
        ]

        nodes_colors = []
        nodes_display_labels = {}
        for index, class_ in enumerate(self.data_nodes_only.nodes_classes):
            if index in atoms_nodes_indices:
                symbol = self.atoms_classes_symbols[class_.item()]
                if supergraph:
                    nodes_colors.append("black")
                elif symbol == "None":
                    nodes_colors.append("white")
                else:
                    nodes_colors.append(atoms_colors[symbol])
                if simple_display:
                    nodes_display_labels[index] = symbol
                elif supergraph:
                    nodes_display_labels[index] = "a"
                else:
                    nodes_display_labels[index] = symbol + ":" + str(index)

            elif index in bonds_nodes_indices:
                type = self.bonds_classes_types[class_.item()]
                if supergraph:
                    nodes_colors.append("lightgrey")
                elif type == "None":
                    nodes_colors.append("lightgrey")
                else:
                    nodes_colors.append(bonds_colors[type])
                if simple_display:
                    if type == "None":
                        nodes_display_labels[index] = "n"
                    else:
                        nodes_display_labels[index] = type[0]
                elif supergraph:
                    nodes_display_labels[index] = "b"
                else:
                    nodes_display_labels[index] = type[:3] + ":" + str(index)

        # Set bonds colors and display labels
        edges_colors = []
        edges_display_labels = {}
        for edge in networkx_graph.edges(data=True):
            edges_colors.append("grey")
            edges_display_labels[(edge[0], edge[1])] = " "

        if simple_display and (not large_molecule_display):
            # Display graph
            nx.draw(
                networkx_graph,
                positions,
                edge_color=edges_colors,
                width=5,
                linewidths=1,
                node_size=1000,
                node_color=nodes_colors,
                alpha=0.7,
                ax=axis,
            )
            nx.draw_networkx_labels(
                networkx_graph,
                positions,
                labels=nodes_display_labels,
                font_color="white",
                font_size=20,
                ax=axis,
                font_weight="bold",
            )
            nx.draw_networkx_edge_labels(
                networkx_graph,
                positions,
                edge_labels=edges_display_labels,
                font_color="black",
                font_size=7,
                ax=axis,
                bbox=dict(facecolor="red", alpha=0),
            )
        elif simple_display and large_molecule_display:
            nx.draw(
                networkx_graph,
                positions,
                edge_color=edges_colors,
                width=4,
                linewidths=1,
                node_size=125,
                node_color=nodes_colors,
                alpha=0.7,
                ax=axis,
            )
            nx.draw_networkx_labels(
                networkx_graph,
                positions,
                labels=nodes_display_labels,
                font_color="white",
                font_size=9,
                ax=axis,
                font_weight="bold",
            )
            nx.draw_networkx_edge_labels(
                networkx_graph,
                positions,
                edge_labels=edges_display_labels,
                font_color="black",
                font_size=2,
                ax=axis,
                bbox=dict(facecolor="red", alpha=0),
            )

        else:
            nx.draw(
                networkx_graph,
                positions,
                edge_color=edges_colors,
                width=10,
                linewidths=1,
                node_size=500,
                node_color=nodes_colors,
                alpha=0.3,
                ax=axis,
            )
            nx.draw_networkx_labels(
                networkx_graph,
                positions,
                labels=nodes_display_labels,
                font_color="black",
                font_size=10,
                ax=axis,
            )
            nx.draw_networkx_edge_labels(
                networkx_graph,
                positions,
                edge_labels=edges_display_labels,
                font_color="black",
                font_size=7,
                ax=axis,
                bbox=dict(facecolor="red", alpha=0),
            )

    def from_predictions_nodes_only(
        self,
        predicted_atoms_classes,
        predicted_bonds_classes,
        data_nodes_only,
        remove_none_predictions=True,
    ):

        atoms_indices = [
            i for i, type in enumerate(data_nodes_only.nodes_types) if type == 1
        ]
        bonds_indices = [
            i for i, type in enumerate(data_nodes_only.nodes_types) if type == 0
        ]

        removed_atom_indices = []
        # Read nodes
        for atom_index, atom_class in zip(atoms_indices, predicted_atoms_classes):
            if remove_none_predictions and (
                (self.atoms_classes_symbols[atom_class] == "None")
                or (self.atoms_classes_symbols[atom_class] == "Decoy")
            ):
                # Break atoms "index" field, but it is not a problem.
                removed_atom_indices.append(atom_index)
                continue

            self.nodes.append(
                {
                    "index": atom_index,
                    "class": atom_class,
                    "position": data_nodes_only.nodes_positions[atom_index].tolist(),
                    "type": 1,
                }
            )
            self.atoms.append(
                {
                    "index": atom_index,
                    "class": atom_class,
                    "position": data_nodes_only.nodes_positions[atom_index].tolist(),
                    "type": 1,
                }
            )

        # Get bonds neighboring atoms indices
        neighboring_atoms_indices = defaultdict(list)
        for index_b, index_e in data_nodes_only.edge_index.t().tolist():
            if data_nodes_only.nodes_types[index_b] == 0:
                if index_e not in neighboring_atoms_indices[index_b]:
                    neighboring_atoms_indices[index_b].append(index_e)
            if data_nodes_only.nodes_types[index_e] == 0:
                if index_b not in neighboring_atoms_indices[index_e]:
                    neighboring_atoms_indices[index_e].append(index_b)

        for index, (bond_index, bond_class) in enumerate(
            zip(bonds_indices, predicted_bonds_classes)
        ):
            # Remove bonds predicted as None
            if remove_none_predictions and (
                (self.bonds_classes_types[bond_class] == "None")
                or self.bonds_classes_types[bond_class] == "Decoy"
            ):
                continue

            self.nodes.append(
                {
                    "index": bond_index,
                    "class": bond_class,
                    "position": data_nodes_only.nodes_positions[bond_index].tolist(),
                    "type": 0,
                }
            )

            if remove_none_predictions and len(removed_atom_indices):
                # Remove bonds connected to removed atoms
                b, e = neighboring_atoms_indices[bond_index]
                if b in removed_atom_indices or e in removed_atom_indices:
                    continue
                # Shift remaining atoms indices
                b -= sum(
                    [
                        b > removed_atom_index
                        for removed_atom_index in removed_atom_indices
                    ]
                )
                e -= sum(
                    [
                        e > removed_atom_index
                        for removed_atom_index in removed_atom_indices
                    ]
                )

            self.bonds.append(
                {
                    "index": neighboring_atoms_indices[bond_index],
                    "class": bond_class,
                    "position": data_nodes_only.nodes_positions[bond_index].tolist(),
                    "type": 0,
                }
            )

        if not remove_none_predictions:
            for bond_index, (index_b, index_e) in enumerate(
                data_nodes_only.edge_index.t().tolist()
            ):
                self.edges.append({"index": [index_b, index_e]})

            self.data_nodes_only = Data(
                x=torch.tensor([[0] for node in self.nodes]),
                edge_index=torch.tensor([edge["index"] for edge in self.edges])
                .t()
                .to(torch.long)
                .view(2, -1),
                nodes_positions=torch.tensor(
                    [node["position"] for node in self.nodes]
                ).to(torch.long),
                nodes_classes=torch.tensor([node["class"] for node in self.nodes]),
                nodes_types=torch.tensor([node["type"] for node in self.nodes]),
            )

            return self

        if remove_none_predictions:
            # Recompute nodes, edges
            self.set_nodes_edges_nodes_only()
            data = self.to_torch_nodes_only()
            return self

    def get_data(self):
        return self.data

    def get_neighbors(self, atom):
        neighbors = []
        for bond in self.bonds:
            b, e = bond["index"]
            if atom["index"] == b:
                neighbors.append(self.atoms[e])
            if atom["index"] == e:
                neighbors.append(self.atoms[b])
        return neighbors

    def get_neighboring_bonds_from_atom(self, atom_index):
        neighbors = []
        for bond in self.bonds:
            if (atom_index == bond["index"][0]) or (atom_index == bond["index"][1]):
                neighbors.append(bond)
        return neighbors

    def add_rdkit_atoms(self, molecule):
        for atom in self.atoms:
            atom_symbol = self.atoms_classes_symbols[atom["class"]]

            # Atom with charges
            if "," in atom_symbol:
                atom_symbol, charge = atom_symbol.split(",")
                rdkit_atom = Chem.Atom(atom_symbol)
                rdkit_atom.SetFormalCharge(int(charge))
                molecule.AddAtom(rdkit_atom)
                continue

            # Abbreviation
            if atom_symbol == "R":
                new_atom = Chem.Atom("C")
                new_atom.SetProp("atomLabel", "abb")
                molecule.AddAtom(new_atom)
                continue

            molecule.AddAtom(Chem.Atom(atom_symbol))
        return molecule

    def add_rdkit_bonds(self, molecule, assign_stereo):
        map = {
            "SINGLE": Chem.rdchem.BondType.SINGLE,
            "DOUBLE": Chem.rdchem.BondType.DOUBLE,
            "TRIPLE": Chem.rdchem.BondType.TRIPLE,
            "AROMATIC": Chem.rdchem.BondType.AROMATIC,
            "DASHED": Chem.rdchem.BondType.SINGLE,
            "SOLID": Chem.rdchem.BondType.SINGLE,
        }
        needs_update_stereo = False
        bonds_directions = [[], []]
        for ib, bond in enumerate(self.bonds):
            if assign_stereo and isinstance(bond["class"], str):
                bond_type = bond["class"]
                if bond_type == "DASHED-UP":
                    begin_index = bond["index"][0]
                    end_index = bond["index"][1]
                    molecule.AddBond(begin_index, end_index, map["SINGLE"])
                    bonds_directions[0].append((begin_index, end_index))
                    bonds_directions[1].append("DASHED")
                    needs_update_stereo = True
                elif bond_type == "DASHED-DOWN":
                    begin_index = bond["index"][0]
                    end_index = bond["index"][1]
                    molecule.AddBond(end_index, begin_index, map["SINGLE"])
                    bonds_directions[0].append((end_index, begin_index))
                    bonds_directions[1].append("DASHED")
                    needs_update_stereo = True
                elif bond_type == "SOLID-UP":
                    begin_index = bond["index"][0]
                    end_index = bond["index"][1]
                    molecule.AddBond(begin_index, end_index, map["SINGLE"])
                    bonds_directions[0].append((begin_index, end_index))
                    bonds_directions[1].append("SOLID")
                    self.needs_update_stereo = True
                elif bond_type == "SOLID-DOWN":
                    begin_index = bond["index"][0]
                    end_index = bond["index"][1]
                    molecule.AddBond(end_index, begin_index, map["SINGLE"])
                    bonds_directions[0].append((end_index, begin_index))
                    bonds_directions[1].append("SOLID")
                    needs_update_stereo = True
            elif isinstance(bond["class"], str):
                begin_index = bond["index"][0]
                end_index = bond["index"][1]
                molecule.AddBond(begin_index, end_index, map["SINGLE"])
            else:
                bond_type = self.bonds_classes_types[bond["class"]]
                begin_index = bond["index"][0]
                end_index = bond["index"][1]
                molecule.AddBond(begin_index, end_index, map[bond_type])

        return molecule, needs_update_stereo, bonds_directions

    def get_atom_with_rdkit_idx(self, index):
        return self.atoms[index], index

    def remove_atom_with_rdkit_idx(self, atom_index_rdkit):
        # Get graph atom
        _, atom_index = self.get_atom_with_rdkit_idx(atom_index_rdkit)

        # Remove atom
        self.atoms = [
            atom
            for atom_index, atom in enumerate(self.atoms)
            if atom_index != atom_index_rdkit
        ]

        # Update atoms
        for i, atom in enumerate(self.atoms):
            self.atoms[i]["index"] -= int(atom["index"] > atom_index_rdkit)

        # Remove bonds
        bond_remove_indices = []
        for i, bond in enumerate(self.bonds):
            if atom_index_rdkit in bond["index"]:
                bond_remove_indices.append(i)
        self.bonds = [
            bond
            for bond_index, bond in enumerate(self.bonds)
            if not (bond_index in bond_remove_indices)
        ]

        # Update bonds
        for i, bond in enumerate(self.bonds):
            self.bonds[i]["index"][0] -= int(bond["index"][0] > atom_index_rdkit)
            self.bonds[i]["index"][1] -= int(bond["index"][1] > atom_index_rdkit)

    def needs_abbreviations_detection(self):
        for atom in self.atoms:
            atom_symbol = self.atoms_classes_symbols[atom["class"]]
            if atom_symbol == "R":
                return True
        return False

    def to_rdkit(
        self,
        abbreviations,
        abbreviations_smiles_mapping,
        ocr_atoms_classes_mapping,
        spelling_corrector,
        align_rdkit_output=False,
        assign_stereo=False,
        remove_hydrogens=False,
        postprocessing_flags=None,
    ):
        if postprocessing_flags is None:
            postprocessing_flags = {}

        graph_postprocessor = GraphPostprocessor(
            self,
            abbreviations,
            abbreviations_smiles_mapping,
            ocr_atoms_classes_mapping,
            spelling_corrector,
            postprocessing_flags=postprocessing_flags,
            align_rdkit_output=align_rdkit_output,
            assign_stereo=assign_stereo,
            remove_hydrogens=remove_hydrogens,
            filename_logging=self.filename_logging,
        )
        # Post-processed graph
        graph_postprocessor.postprocess_before_rdkit_molecule_creation()

        # Create RDKit molecule
        molecule = Chem.RWMol()
        molecule = self.add_rdkit_atoms(molecule)
        molecule, needs_update_stereo, bonds_directions = self.add_rdkit_bonds(
            molecule, assign_stereo
        )
        molecule = molecule.GetMol()
        graph_postprocessor.molecule = molecule
        graph_postprocessor.needs_update_stereo = needs_update_stereo
        graph_postprocessor.bonds_directions = bonds_directions

        # Post-process RDkit molecule and graph
        molecule = graph_postprocessor.postprocess_after_rdkit_molecule_creation()

        return molecule
