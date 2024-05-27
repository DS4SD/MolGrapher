import os
import math
import json 
from matplotlib import colors as mcolors
import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDepictor
import math
from collections import defaultdict, Counter
import copy
from rdkit.Geometry.rdGeometry import Point3D

from molgrapher.utils.utils_postprocessing import assign_stereo_centers


class MolecularGraph():
    def __init__(self, config_dataset_graph):
        """
        Generic graph structure to ease conversions.
            - Atoms: Molecule's atoms.
            - Bonds: Molecule's bonds
            - Node: Graph's nodes.
            - Edge: Graph's edges.
            - Data: Pytorch Geometric graph structure
        """
        # Core of the class, carry information
        self.atoms = [] # Constraint: atoms indices is a continuous sequence 0, 1, 2, ..., N
        self.bonds = [] # A bond indices [b, e] correspond to positions in the self.atoms list (+1)

        self.nodes = []
        self.edges = []
        self.data_nodes_only = None
        self.data = None
        self.bond_size = 0
        self.config_dataset_graph = config_dataset_graph
        self.superatoms = []
        self.symbols_classes_atoms = json.load(open(os.path.dirname(__file__) + f"/../../data/vocabularies/vocabulary_atoms_{config_dataset_graph['nb_atoms_classes']}.json"))
        self.types_classes_bonds = json.load(open(os.path.dirname(__file__) + f"/../../data/vocabularies/vocabulary_bonds_{config_dataset_graph['nb_bonds_classes']}.json"))
        self.atoms_classes_symbols = {v: k for k,v in self.symbols_classes_atoms.items()}
        self.bonds_classes_types = {v: k for k,v in self.types_classes_bonds.items()}

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

                self.atoms.append({
                    "index": atom_index,
                    "class": self.symbols_classes_atoms[atom_symbol],
                    "position": [keypoints[atom_index][0], keypoints[atom_index][1]],
                    "type": 1
                })
                atom_index += 1

            if isinstance(a["connecting_nodes"], list) or (not math.isnan(a["connecting_nodes"])):
                b, e = a["connecting_nodes"]
                b_i = (annotation[annotation.id == b].index - annotation.index[0])[0] 
                e_i = (annotation[annotation.id == e].index - annotation.index[0])[0] 
                begin_position = [keypoints[b_i][0], keypoints[b_i][1]]
                end_position = [keypoints[e_i][0], keypoints[e_i][1]]

                if bond_index == 0:
                    self.bond_size = math.sqrt((begin_position[0] - end_position[0])**2 + (begin_position[1] - end_position[1])**2)
                    
                bond_type = a["smarts"]
                if bond_type == "-" or bond_type == "up" or bond_type == "down" or bond_type == "/" or bond_type == "\\":
                    bond_type = "SINGLE"
                elif bond_type == "=" or bond_type == "either":
                    bond_type = "DOUBLE" 
                elif bond_type == "#":
                    bond_type = "TRIPLE"
                if bond_type not in self.types_classes_bonds:
                    print(f"Bond type: {bond_type} rejected")
                    bond_type = "None"

                self.bonds.append({
                    "index": [b_i, e_i],
                    "class": self.types_classes_bonds[bond_type],
                    "position": [int((begin_position[0] + end_position[0])/2), int((begin_position[1] + end_position[1])/2)],
                    "type": 0
                })
                bond_index += 1
        return self

    def from_rdkit_molecule(self, molecule, keypoints):
        if molecule is None:
            print("The Molfile can't be read")
            return 
        
        # Remove aromatic bonds (Kekulize)
        #molecule = rdMolDraw2D.PrepareMolForDrawing(molecule, addChiralHs=False)

        # Set wedge bonds
        #rdDepictor.Compute2DCoords(molecule)
        Chem.WedgeMolBonds(molecule, molecule.GetConformers()[0]) 

        # Read atoms
        for atom_index, atom in enumerate(molecule.GetAtoms()):
            if atom.HasProp("_displayLabel"):
                atom.SetAtomicNum(0)

            keypoint = keypoints[atom_index]
            atom_symbol = atom.GetSymbol()
            if atom_symbol == "*":
                atom_symbol = "R"
            elif atom.GetFormalCharge():
                atom_symbol = atom.GetSymbol() + "," + str(atom.GetFormalCharge())
            if atom_symbol not in self.symbols_classes_atoms:
                atom_symbol = "None"

            self.atoms.append({
                "index": atom_index,
                "class": self.symbols_classes_atoms[atom_symbol],
                "position": [keypoint[0], keypoint[1]],
                "type": 1
            })

        # Read bonds
        for bond_index, bond in enumerate(molecule.GetBonds()):
            b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            begin_position = [keypoints[b][0], keypoints[b][1]]
            end_position = [keypoints[e][0], keypoints[e][1]]

            if bond_index == 0:
                self.bond_size = math.sqrt((begin_position[0] - end_position[0])**2 + (begin_position[1] - end_position[1])**2)

            bond_type = str(bond.GetBondType())
            
            if bond_type not in self.types_classes_bonds:
                bond_type = "None"

            if str(bond.GetBondDir()) == "BEGINWEDGE":
                bond_type = "SOLID"
            
            if str(bond.GetBondDir()) == "BEGINDASH":
                bond_type = "DASHED"

            self.bonds.append({
                "index": [b, e],
                "class": self.types_classes_bonds[bond_type],
                "position": [int((begin_position[0] + end_position[0])/2), int((begin_position[1] + end_position[1])/2)],
                "type": 0
            })
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
            #atoms_edge = {"index": [b, e]}
            atom_b_bond_edge = {"index": [b, bond_node["index"]]}
            bond_atom_e_edge = {"index": [bond_node["index"], e]}
            #edges += [atoms_edge, atom_b_bond_edge, bond_atom_e_edge]
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
            x = torch.tensor([[0] for node in self.nodes]),
            edge_index = torch.tensor([edge["index"] for edge in self.edges]).t().to(torch.long).view(2, -1), 
            nodes_positions = torch.tensor([node["position"] for node in self.nodes]).to(torch.long), 
            nodes_classes = torch.tensor([node["class"] for node in self.nodes]), 
            nodes_types = torch.tensor([node["type"] for node in self.nodes]), 
        )
        return self.data_nodes_only
        
    def to_torch(self):
        self.duplicate_edges()
        self.data = Data(
            x = torch.tensor([[0] for node in self.nodes]),
            edge_index = torch.tensor([edge["index"] for edge in self.edges]).t().to(torch.long).view(2, -1), 
            edges_classes = torch.tensor([edge["class"] for edge in self.edges]), 
            nodes_positions = torch.tensor([node["position"] for node in self.nodes]).to(torch.long), 
            nodes_classes = torch.tensor([node["class"] for node in self.nodes]), 
            nodes_types = torch.tensor([node["type"] for node in self.nodes]), 
        )
        return self.data
    
    def display_data(self, axis):
        if self.data is None:
            print("Pytorch geometric graph is not defined")
            return

        # Define mappings
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())*50
        white_colors = ['beige', 'bisque', 'blanchedalmond', 'cornsilk', 'antiquewhite', 'ghostwhite', 'w', 'whitesmoke', 'white', 'snow', 'seashell', 'mistyrose', 'linen', 'floralwhite', 'ivory', 'honeydew', 'mintcream', 'azure', 'aliceblue', 'lavenderblush']
        colors = [color for color in colors if color not in white_colors]*50
        
        atoms_colors = {k: colors[v] for k,v in self.symbols_classes_atoms.items()}
        bonds_colors = {k: colors[v] for k,v in self.types_classes_bonds.items()}
        
        # Convert Pytorch graph to Networkx graph
        networkx_graph = torch_geometric.utils.to_networkx(
            self.data, 
            to_undirected=True,
            edge_attrs=["edges_classes"],
            node_attrs=["nodes_classes"]
        )
            
        # Set nodes positions, colors and display labels
        positions = [[position[0], (self.config_dataset_graph["image_size"][1] - 1) - position[1]] for position in self.data.nodes_positions.tolist()]
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
            edges_display_labels[(edge[0], edge[1])] = edge_type[:3] + ":" + str(atom_index)

        # Display graph
        nx.draw(networkx_graph, positions, edge_color=edges_colors, width=10, linewidths=1, node_size=500, node_color=nodes_colors, alpha=0.3, ax=axis)
        nx.draw_networkx_labels(networkx_graph, positions, labels=nodes_display_labels, font_color='black', font_size=10, ax=axis)
        nx.draw_networkx_edge_labels(networkx_graph, positions, edge_labels=edges_display_labels, font_color='black', font_size=7, ax=axis, bbox=dict(facecolor='red', alpha=0))
        
    def display_data_nodes_only(self, axis, simple_display=False, supergraph=False, large_molecule_display=True):
        if self.data_nodes_only is None:
            print("Pytorch geometric graph is not defined")
            return

        # Define mappings
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())*50
        white_colors = ['beige', 'bisque', 'blanchedalmond', 'cornsilk', 'antiquewhite', 'ghostwhite', 'w', 'whitesmoke', 'white', 'snow', 'seashell', 'mistyrose', 'linen', 'floralwhite', 'ivory', 'honeydew', 'mintcream', 'azure', 'aliceblue', 'lavenderblush']
        colors = [color for color in colors if color not in white_colors]*50
        
        atoms_colors = {k: colors[v] for k,v in self.symbols_classes_atoms.items()}
        bonds_colors = {k: colors[v] for k,v in self.types_classes_bonds.items()}
        
        # Convert Pytorch graph to Networkx graph
        networkx_graph = torch_geometric.utils.to_networkx(
            self.data_nodes_only, 
            to_undirected = True
        )  

        # Set nodes positions, colors and display labels
        positions = [[position[0], ((self.config_dataset_graph["image_size"][1] - 1) - position[1])]for position in self.data_nodes_only.nodes_positions.tolist()]
        atoms_nodes_indices = [i for i, type in enumerate(self.data_nodes_only.nodes_types) if type == 1]
        bonds_nodes_indices = [i for i, type in enumerate(self.data_nodes_only.nodes_types) if type == 0]

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
            nx.draw(networkx_graph, positions, edge_color=edges_colors, width=5, linewidths=1, node_size=1000, node_color=nodes_colors, alpha=0.7, ax=axis)
            nx.draw_networkx_labels(networkx_graph, positions, labels=nodes_display_labels, font_color='white', font_size=20, ax=axis, font_weight="bold")
            nx.draw_networkx_edge_labels(networkx_graph, positions, edge_labels=edges_display_labels, font_color='black', font_size=7, ax=axis, bbox=dict(facecolor='red', alpha=0))
        elif simple_display and large_molecule_display:
            nx.draw(networkx_graph, positions, edge_color=edges_colors, width=4, linewidths=1, node_size=125, node_color=nodes_colors, alpha=0.7, ax=axis)
            nx.draw_networkx_labels(networkx_graph, positions, labels=nodes_display_labels, font_color='white', font_size=9, ax=axis, font_weight="bold")
            nx.draw_networkx_edge_labels(networkx_graph, positions, edge_labels=edges_display_labels, font_color='black', font_size=2, ax=axis, bbox=dict(facecolor='red', alpha=0))
        
        else:
            nx.draw(networkx_graph, positions, edge_color=edges_colors, width=10, linewidths=1, node_size=500, node_color=nodes_colors, alpha=0.3, ax=axis)
            nx.draw_networkx_labels(networkx_graph, positions, labels=nodes_display_labels, font_color='black', font_size=10, ax=axis)
            nx.draw_networkx_edge_labels(networkx_graph, positions, edge_labels=edges_display_labels, font_color='black', font_size=7, ax=axis, bbox=dict(facecolor='red', alpha=0))
        

    def from_predictions_nodes_only(self, predicted_atoms_classes, predicted_bonds_classes, data_nodes_only, remove_none_predictions=True):
    
        atoms_indices = [i for i, type in enumerate(data_nodes_only.nodes_types) if type == 1]
        bonds_indices = [i for i, type in enumerate(data_nodes_only.nodes_types) if type == 0]
       
        removed_atom_indices = []
        # Read nodes
        for atom_index, atom_class in zip(atoms_indices, predicted_atoms_classes):
            if remove_none_predictions and ((self.atoms_classes_symbols[atom_class] == "None") or (self.atoms_classes_symbols[atom_class] == "Decoy")):
                # Break atoms "index" field, but it is not a problem.
                removed_atom_indices.append(atom_index)
                continue
    
            self.nodes.append({
                "index": atom_index,
                "class": atom_class,
                "position": data_nodes_only.nodes_positions[atom_index].tolist(),
                "type": 1
            })
            self.atoms.append({
                "index": atom_index,
                "class": atom_class,
                "position": data_nodes_only.nodes_positions[atom_index].tolist(),
                "type": 1
            })

        # Get bonds neighboring atoms indices
        neighboring_atoms_indices = defaultdict(list)
        for index_b, index_e in data_nodes_only.edge_index.t().tolist():
            if data_nodes_only.nodes_types[index_b] == 0: 
                    if index_e not in neighboring_atoms_indices[index_b]:
                        neighboring_atoms_indices[index_b].append(index_e)
            if data_nodes_only.nodes_types[index_e] == 0: 
                if index_b not in neighboring_atoms_indices[index_e]:
                    neighboring_atoms_indices[index_e].append(index_b)

        for index, (bond_index, bond_class) in enumerate(zip(bonds_indices, predicted_bonds_classes)):
            # Remove bonds predicted as None
            if remove_none_predictions and ((self.bonds_classes_types[bond_class] == "None") or self.bonds_classes_types[bond_class] == "Decoy"):
                continue

            self.nodes.append({
                "index": bond_index,
                "class": bond_class,
                "position": data_nodes_only.nodes_positions[bond_index].tolist(),
                "type": 0
            })

            if remove_none_predictions and len(removed_atom_indices):
                # Remove bonds connected to removed atoms
                b, e = neighboring_atoms_indices[bond_index]
                if b in removed_atom_indices or e in removed_atom_indices:
                    continue
                # Shift remaining atoms indices
                b -= sum([b > removed_atom_index for removed_atom_index in removed_atom_indices])
                e -= sum([e > removed_atom_index for removed_atom_index in removed_atom_indices])

            self.bonds.append({
                "index": neighboring_atoms_indices[bond_index],
                "class": bond_class,
                "position": data_nodes_only.nodes_positions[bond_index].tolist(),
                "type": 0
            })
        
        if not remove_none_predictions:
            for bond_index, (index_b, index_e) in enumerate(data_nodes_only.edge_index.t().tolist()):     
                self.edges.append({"index": [index_b, index_e]})
                
            self.data_nodes_only = Data(
                x = torch.tensor([[0] for node in self.nodes]),
                edge_index = torch.tensor([edge["index"] for edge in self.edges]).t().to(torch.long).view(2, -1), 
                nodes_positions = torch.tensor([node["position"] for node in self.nodes]).to(torch.long), 
                nodes_classes = torch.tensor([node["class"] for node in self.nodes]), 
                nodes_types = torch.tensor([node["type"] for node in self.nodes])
            )

            return self
        
        if remove_none_predictions:
            # Recompute nodes, edges
            self.set_nodes_edges_nodes_only()
            data = self.to_torch_nodes_only()
            return self

    def needs_abbreviations_detection(self):
        for atom in self.atoms:
            atom_symbol = self.atoms_classes_symbols[atom["class"]]
            if atom_symbol == "R":
                return True
        return False

    def try_sanitize_molecule(self, molecule):
        try:
            Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return molecule
        except:
            print("The predicted molecule can not be sanitized")
            return False

    def to_rdkit(
        self, 
        abbreviations, 
        abbreviations_smiles_mapping, 
        ocr_atoms_classes_mapping, 
        spelling_corrector, 
        stop=False, 
        align_rdkit_output=False, 
        assign_stereo=False
        ):

        molecule = Chem.RWMol()
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
            
        map = {
            "SINGLE": Chem.rdchem.BondType.SINGLE,
            "DOUBLE": Chem.rdchem.BondType.DOUBLE,
            "TRIPLE": Chem.rdchem.BondType.TRIPLE,
            "AROMATIC": Chem.rdchem.BondType.AROMATIC,
            "DASHED": Chem.rdchem.BondType.SINGLE,
            "SOLID": Chem.rdchem.BondType.SINGLE
        }
        
        need_update_stereo = False
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
                    need_update_stereo = True

                elif bond_type == "DASHED-DOWN":
                    begin_index = bond["index"][0]
                    end_index = bond["index"][1]
                    molecule.AddBond(end_index, begin_index, map["SINGLE"])

                    bonds_directions[0].append((end_index, begin_index))
                    bonds_directions[1].append("DASHED")
                    need_update_stereo = True
                
                elif bond_type == "SOLID-UP":
                    begin_index = bond["index"][0]
                    end_index = bond["index"][1]
                    molecule.AddBond(begin_index, end_index, map["SINGLE"])

                    bonds_directions[0].append((begin_index, end_index))
                    bonds_directions[1].append("SOLID")
                    need_update_stereo = True

                elif bond_type == "SOLID-DOWN":
                    begin_index = bond["index"][0]
                    end_index = bond["index"][1]
                    molecule.AddBond(end_index, begin_index, map["SINGLE"])

                    bonds_directions[0].append((end_index, begin_index))
                    bonds_directions[1].append("SOLID")
                    need_update_stereo = True

            elif isinstance(bond["class"], str):    
                begin_index = bond["index"][0]
                end_index = bond["index"][1]
                molecule.AddBond(begin_index, end_index, map["SINGLE"])

            else:
                bond_type = self.bonds_classes_types[bond["class"]]
                begin_index = bond["index"][0]
                end_index = bond["index"][1]
                molecule.AddBond(begin_index, end_index, map[bond_type])

        molecule = molecule.GetMol()

        # Post-process aromatic bonds not in cycles
        for bond in molecule.GetBonds():
            try:
                if (str(bond.GetBondType()) == "AROMATIC") and (not bond.IsInRing()):
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetIsAromatic(False)
                    molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).SetIsAromatic(False)
            except:
                print("Error in aromatic post-processing")
                pass 
            
        # Post-process aromatic rings
        ringInfo = molecule.GetRingInfo()
        bonds_rings = ringInfo.BondRings()
        for bonds_ring in bonds_rings:
            types = []
            for bond in bonds_ring:
                types.append(str(molecule.GetBondWithIdx(bond).GetBondType()))

            most_common_type = Counter(types).most_common(1)[0][0]
            if (most_common_type == "AROMATIC") and any([type != "AROMATIC" for type in types]):
                for bond in bonds_ring:
                    molecule.GetBondWithIdx(bond).SetBondType(Chem.rdchem.BondType.AROMATIC)
        
        if align_rdkit_output:
            molecule_aligned = copy.deepcopy(molecule)
            rdDepictor.Compute2DCoords(molecule_aligned) #rdMolDraw2D.PrepareMolForDrawing(molecule_aligned, addChiralHs=False)
            for atom_index, atom in enumerate(self.atoms):
                position = Point3D()
                position.x, position.y, position.z = atom["position"][0]/64, -atom["position"][1]/64, 0
                molecule_aligned.GetConformer(0).SetAtomPosition(atom_index, position)

        # Return the molecule if it does not contain any abbreviations and is sanitized
        if all([(self.atoms_classes_symbols[atom["class"]] != "R") for atom in self.atoms]) and len(Chem.GetMolFrags(molecule)) == 1:
            if align_rdkit_output:
                rdDepictor.GenerateDepictionMatching2DStructure(
                    molecule,
                    reference = molecule_aligned
                )
                try:
                    rdDepictor.GenerateDepictionMatching2DStructure(
                        molecule,
                        reference = molecule_aligned
                    )
                except:
                    print("Aligning RDKit molecule failed")
                    return Chem.MolFromSmiles("C")
            if assign_stereo and need_update_stereo:
                # Assign stereo-chemistry
                molecule = assign_stereo_centers(molecule, bonds_directions)
                
            molecule_return = self.try_sanitize_molecule(molecule)
            if molecule_return:
                return molecule_return
            
        original_molecule = copy.deepcopy(molecule)
        original_abbreviations = copy.deepcopy(abbreviations)
        matched_abbreviations_indices = []

        # Resolve abbreviations
        for abbreviation_index, atom in enumerate(self.atoms): 
            if self.atoms_classes_symbols[atom["class"]] == "R":
                for i, abbreviation in enumerate(abbreviations):
                    if (atom["position"][0] >= (abbreviation["box"][0][0])) and \
                        (atom["position"][0] <= (abbreviation["box"][1][0])) and \
                        (atom["position"][1] >= (abbreviation["box"][0][1])) and \
                        (atom["position"][1] <= (abbreviation["box"][1][1])):
                    
                        # Remove the matched abbreviation
                        abbreviations = [abb for i_abb, abb in enumerate(abbreviations) if (i_abb != i)]
                        
                        if (abbreviation["text"] not in abbreviations_smiles_mapping) and (abbreviation["text"] not in ocr_atoms_classes_mapping):
                            #print("Before correction: ", abbreviation["text"])
                            self.superatoms.append(abbreviation["text"])
                            abbreviation["text"] = spelling_corrector(abbreviation["text"])
                            #print("After correction: ", abbreviation["text"])
                            
                            Chem.SetAtomAlias(molecule.GetAtomWithIdx(abbreviation_index), f"{abbreviation['text']}")

                        if abbreviation["text"] in ocr_atoms_classes_mapping:
                            #print(f"The OCR detection {abbreviation['text']} is replaced in the molecule.")
                            molecule_editable = Chem.EditableMol(molecule)
                            atom_symbol = ocr_atoms_classes_mapping[abbreviation["text"]]["symbol"]

                            # Atom with charge
                            if "," in atom_symbol:
                                atom_symbol, charge = atom_symbol.split(",")
                                rdkit_atom = Chem.Atom(atom_symbol)
                                rdkit_atom.SetFormalCharge(int(charge))
                                molecule_editable.ReplaceAtom(abbreviation_index, rdkit_atom)
                            else:
                                molecule_editable.ReplaceAtom(abbreviation_index, Chem.Atom(atom_symbol))
                            molecule = molecule_editable.GetMol()
                            continue

                        if abbreviation["text"] not in abbreviations_smiles_mapping:
                            molecule.GetAtomWithIdx(abbreviation_index).SetProp("atomLabel", f"[{abbreviation['text']}]") 
                            Chem.SetAtomAlias(molecule.GetAtomWithIdx(abbreviation_index), f"[{abbreviation['text']}]")
                            continue
                    
                        molecule_abbreviation = Chem.MolFromSmiles(abbreviations_smiles_mapping[abbreviation["text"]]["smiles"])

                        if molecule_abbreviation is None:
                            print(f"Problem with abbreviation: {abbreviation}")
                            continue

                        # Save molecule abbreviation connection points
                        multiple_bonds_error = False
                        molecule_abbreviation_connection_points = {}
                        connection_point_abbreviation_index = 0
                        for atom_index, connection_atom in enumerate(molecule_abbreviation.GetAtoms()):
                            if connection_atom.GetSymbol() == "*": 
                                bonds = connection_atom.GetBonds()
                                
                                if len(bonds) > 1:
                                    print("Error: connection atom from abbreviation has multiple bonds")
                                    multiple_bonds_error = True
                                    break

                                bond = bonds[0]
                                # Save connection atoms
                                for neighbor_index in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                                    if molecule_abbreviation.GetAtomWithIdx(neighbor_index).HasProp(f"{abbreviation_index}-attachmentIndex"):
                                        # Different atoms of the molecule can attach to the same position in the abbreviation
                                        molecule_abbreviation.GetAtomWithIdx(neighbor_index).SetProp(
                                            f"{abbreviation_index}-attachmentIndex",
                                            molecule_abbreviation.GetAtomWithIdx(neighbor_index).GetProp(f"{abbreviation_index}-attachmentIndex") + "," + str(connection_point_abbreviation_index)
                                        )
                                    else:
                                        molecule_abbreviation.GetAtomWithIdx(neighbor_index).SetProp(f"{abbreviation_index}-attachmentIndex", str(connection_point_abbreviation_index))
                                    molecule_abbreviation_connection_points[str(connection_point_abbreviation_index)] = {
                                        "bond": bond.GetBondType()
                                    }

                                connection_point_abbreviation_index += 1

                        if multiple_bonds_error:
                            continue

                        nb_connection_points_abbreviation_molecule = connection_point_abbreviation_index
                        
                        # Remove extra atoms in molecule abbreviation
                        molecule_abbreviation_editable = Chem.EditableMol(molecule_abbreviation)
                        for atom_index in range(molecule_abbreviation.GetNumAtoms()-1, -1, -1):
                            if molecule_abbreviation.GetAtomWithIdx(atom_index).GetSymbol() == "*": 
                                molecule_abbreviation_editable.RemoveAtom(atom_index)
                        molecule_abbreviation = molecule_abbreviation_editable.GetMol()

                        # Retrieve connection points indices
                        for atom_index, connection_atom in enumerate(molecule_abbreviation.GetAtoms()):
                            if connection_atom.HasProp(f"{abbreviation_index}-attachmentIndex"):
                                connection_point_abbreviation_indices = connection_atom.GetProp(f"{abbreviation_index}-attachmentIndex").split(",")
                                for connection_point_abbreviation_index in connection_point_abbreviation_indices:
                                    molecule_abbreviation_connection_points[connection_point_abbreviation_index]["index"] = atom_index
                            
                        # Save molecule connection points
                        connection_point_index = 0
                        for connection_atom in molecule.GetAtomWithIdx(abbreviation_index).GetNeighbors():
                            connection_point_index += 1

                        nb_connection_points_molecule = connection_point_index

                        if nb_connection_points_molecule != nb_connection_points_abbreviation_molecule:
                            print(f"The number of connection point between the predicted abbreviation node and the associated sub-molecule ({abbreviation['text']}) mismatch")
                            break

                        # Save molecule connection points
                        connection_point_index = 0
                        for connection_atom in molecule.GetAtomWithIdx(abbreviation_index).GetNeighbors():
                            connection_atom.SetProp(f"{abbreviation_index}-attachmentIndex", str(connection_point_index))
                            connection_point_index += 1

                        # Remove abbreviation in molecule
                        matched_abbreviations_indices.append(abbreviation_index)
                        
                        # Retrieve connection points indices
                        molecule_connection_points = {}
                        for atom_index, connection_atom in enumerate(molecule.GetAtoms()):
                            if connection_atom.HasProp(f"{abbreviation_index}-attachmentIndex"):
                                connection_point_index = connection_atom.GetProp(f"{abbreviation_index}-attachmentIndex")
                                molecule_connection_points[connection_point_index] = {
                                    "index": atom_index
                                }

                        offset = molecule.GetNumAtoms()
                        # Combine
                        molecule = Chem.CombineMols(molecule, molecule_abbreviation) 
                        molecule_editable = Chem.EditableMol(molecule)

                        # Add bonds (For multiple connection point abbreviations, the order should be from left to right, by checking bonds positions #TODO)
                        for connection_point_index in molecule_connection_points.keys():
                            try:
                                molecule_editable.AddBond(
                                    molecule_connection_points[connection_point_index]["index"], 
                                    molecule_abbreviation_connection_points[connection_point_index]["index"] + offset, 
                                    order=molecule_abbreviation_connection_points[connection_point_index]["bond"]
                                )
                            except:
                                print('error')
                                print(connection_point_index)
                                print(abbreviation)
                           
                        molecule = molecule_editable.GetMol()

                        break
        
        # Remove abbreviation in molecule (in decreasing order)
        molecule_editable = Chem.EditableMol(molecule)
        for abbreviation_index in sorted(matched_abbreviations_indices, reverse=True):
            molecule_editable.RemoveAtom(abbreviation_index)
        molecule = molecule_editable.GetMol()

        if assign_stereo:
            # Adjust wedge bonds indices after removing "abbreviation connections" atoms
            for bi in range(len(bonds_directions[0])):
                b, e = bonds_directions[0][bi]
                b -= sum([b > i for i in matched_abbreviations_indices])
                e -= sum([e > i for i in matched_abbreviations_indices])
                bonds_directions[0][bi] = b, e

        if assign_stereo and need_update_stereo:
            # Assign stereo-chemistry
            molecule = assign_stereo_centers(molecule, bonds_directions)

        if len(Chem.GetMolFrags(molecule)) == 1:
            if align_rdkit_output:
                molecule_aligned_2 = Chem.EditableMol(molecule_aligned)
                for abbreviation_index in sorted(matched_abbreviations_indices, reverse=True):
                    molecule_aligned_2.RemoveAtom(abbreviation_index)
                molecule_aligned_2 = molecule_aligned_2.GetMol()
                try:
                    rdDepictor.GenerateDepictionMatching2DStructure(
                        molecule,
                        reference = molecule_aligned_2
                    )
                except:
                    print("Aligning RDKit molecule failed")
                    return Chem.MolFromSmiles("C")

            molecule_return = self.try_sanitize_molecule(molecule)
            if molecule_return:
                return molecule_return
        
        elif (not stop) and (len(original_abbreviations) > 0):
            
            for i, abbreviation in enumerate(original_abbreviations):
                
                atoms_matched = []
                for abbreviation_index, atom in enumerate(self.atoms): 
                    if (atom["position"][0] >= (abbreviation["box"][0][0])) and \
                        (atom["position"][0] <= (abbreviation["box"][1][0])) and \
                        (atom["position"][1] >= (abbreviation["box"][0][1])) and \
                        (atom["position"][1] <= (abbreviation["box"][1][1])):
                        atoms_matched.append(atom)

                if len(atoms_matched) > 1:
                    # If multiple atoms are located to the same ocr prediction location, trust the ocr. 
                    # An alternative would be to look at the atoms predictions confidences
                    if abbreviation["text"] in abbreviations_smiles_mapping:
                        atom_class = self.symbols_classes_atoms["R"]
                    elif all(atom["class"] == atoms_matched[0]["class"] for atom in atoms_matched):
                        atom_class = atoms_matched[0]["class"]
                    else:
                        # Get most represented prediction? most popular class?
                        atom_class = self.symbols_classes_atoms["R"] 
                    
                    # Create new atom
                    new_atom_index = len(self.atoms) 
                    self.atoms.append({
                        "index": new_atom_index,
                        "class": atom_class,
                        "position": atoms_matched[0]["position"],
                        "type": 1
                    })

                    # Update previous bonds
                    remove_bonds_indices = []
                    atoms_matched_indices = [atom_index for atom_index, atom in enumerate(self.atoms) if atom in atoms_matched]
                    for bond_index, bond in enumerate(self.bonds):
                        if (bond["index"][0] in atoms_matched_indices) and (bond["index"][1] in atoms_matched_indices):
                            remove_bonds_indices.append(bond_index)

                        if bond["index"][0] in atoms_matched_indices:
                            new_index = [new_atom_index, bond["index"][1]]
                            if all((new_index != bond["index"]) for bond in self.bonds) and all((list(reversed(new_index)) != bond["index"]) for bond in self.bonds):
                                bond["index"] = new_index
                            else:
                                remove_bonds_indices.append(bond_index)
                            continue

                        if bond["index"][1] in atoms_matched_indices:
                            new_index = [bond["index"][0], new_atom_index]
                            if all((new_index != bond["index"]) for bond in self.bonds) and all((list(reversed(new_index)) != bond["index"]) for bond in self.bonds):
                                bond["index"] = new_index
                            else:
                                remove_bonds_indices.append(bond_index)

                    self.bonds = [bond for bond_index, bond in enumerate(self.bonds) if bond_index not in remove_bonds_indices]
               
                    # Remove matched atoms
                    for atom_matched in atoms_matched:
                        self.atoms.remove(atom_matched)
                    
                    # Update bond indices
                    for bond in self.bonds:
                        b = bond["index"][0]
                        e = bond["index"][1]
                        b -= sum([b > removed_atom_index for removed_atom_index in atoms_matched_indices])
                        e -= sum([e > removed_atom_index for removed_atom_index in atoms_matched_indices])
                        bond["index"] = [b, e]

            print("Recursive molecule creation")
            return self.to_rdkit(
                original_abbreviations, 
                abbreviations_smiles_mapping, 
                ocr_atoms_classes_mapping, 
                spelling_corrector, 
                stop=True
            )
            
        if len(Chem.GetMolFrags(original_molecule)) == 1:
            if align_rdkit_output:
                molecule_aligned_2 = Chem.EditableMol(molecule_aligned)
                for abbreviation_index in sorted(matched_abbreviations_indices, reverse=True):
                    molecule_aligned_2.RemoveAtom(abbreviation_index)
                molecule_aligned_2 = molecule_aligned_2.GetMol()
                try:
                    rdDepictor.GenerateDepictionMatching2DStructure(
                        molecule,
                        reference = molecule_aligned_2
                    )
                except:
                    print("Aligning RDKit molecule failed")
                    return Chem.MolFromSmiles("C")

            original_molecule_return = self.try_sanitize_molecule(original_molecule)
            if original_molecule_return:
                return original_molecule_return
        
        # Remove isolated atom
        if (len(Chem.GetMolFrags(original_molecule))) > 1 and (not stop):
            atoms_involved_in_connections = list(set([bond["index"][0] for bond in self.bonds] + [bond["index"][1] for bond in self.bonds]))
            removed_atom_indices = []
            for atom_idx in range(len(self.atoms)):
                if atom_idx not in atoms_involved_in_connections:
                    removed_atom_indices.append(atom_idx)
            removed_atom_indices = sorted(removed_atom_indices, reverse = True)
            #print("Atom indices to remove:" + str(removed_atom_indices))
            if len(removed_atom_indices):
                for bond_idx in range(len(self.bonds)):
                    # Remove bonds connected to removed atoms
                    b, e = self.bonds[bond_idx]["index"]
                    if b in removed_atom_indices or e in removed_atom_indices:
                        continue
                    # Shift remaining atoms indices
                    b -= sum([b > removed_atom_index for removed_atom_index in removed_atom_indices])
                    e -= sum([e > removed_atom_index for removed_atom_index in removed_atom_indices])
                    self.bonds[bond_idx]["index"] = [b, e]
                    
                atoms_new = []
                for atom_idx in range(len(self.atoms)):
                    if atom_idx not in removed_atom_indices:
                        atoms_new.append({
                        "index": self.atoms[atom_idx]["index"],
                        "class": self.atoms[atom_idx]["class"],
                        "position": self.atoms[atom_idx]["position"],
                        "type": 1
                    })
                self.atoms = atoms_new
            return self.to_rdkit(
                original_abbreviations, 
                abbreviations_smiles_mapping, 
                ocr_atoms_classes_mapping, 
                spelling_corrector, 
                stop=True
            )
        
        # TODO Return largest fragment
        return Chem.MolFromSmiles("C")

    def get_data(self):
        return self.data


