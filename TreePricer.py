import numpy as np
from Node import Node
from PricingModel import Engine  # Importation de la classe mère
from datetime import date
from typing import Tuple, List,Union
from Market import Market
from Option import Option

class TreeModel(Engine):
    def __init__(self, market :Market, option:Option, pricing_date: date, n_steps: int, THRESHOLD: float = 1e-7) -> None:
        super().__init__(market, option, pricing_date, n_steps=n_steps)
        self.THRESHOLD = THRESHOLD
        self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.dt))
        self.proba_down = (np.exp(self.market.sigma ** 2 * self.dt) - 1) \
                          / ((1 - self.alpha) * (self.alpha ** (-2) - 1))
        self.proba_up = self.proba_down / self.alpha
        self.proba_mid = 1 - self.proba_down - self.proba_up
        self.root = Node(self.market.S0)
        self.root.proba = 1
        self.tree_price = None
        self.build_tree()

    def is_div_date(self, step: int) -> bool:
        return step < self.t_div <= (step + 1) if self.t_div is not None else False

    def forward(self, parent: Node, step: int) -> float:
        if self.is_div_date(step):
            return parent.S * np.exp(self.market.r * self.dt) - self.market.dividend
        return parent.S * np.exp(self.market.r * self.dt)
    # Get the probabilities for up, mid, down
    def get_proba(self, div_node: Node, step: int) -> Tuple[float, float, float]: 
        average = self.forward(div_node, step)
        var = div_node.S ** 2 * np.exp(2 * self.market.r * self.dt) * (np.exp(self.market.sigma ** 2 * self.dt) - 1)

        proba_down = (div_node.child_mid.S ** (-2) * (var + average ** 2) - 1 - (self.alpha + 1) *
                      (div_node.child_mid.S ** (-1) * average - 1)) / ((1 - self.alpha) * (self.alpha ** (-2) - 1))
        proba_up = (div_node.child_mid.S ** (-1) * average - 1 - (self.alpha ** (-1) - 1) *
                    proba_down) / (self.alpha - 1)
        proba_mid = 1 - proba_down - proba_up
        return proba_up, proba_mid, proba_down

    # Build the upper Nodes of the column
    def build_up(self, trunc_parent: Node, trunc_mid: Node, trunc_up: Node, trunc_down: Node) -> None:
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        while parent.bro_up is not None:
            parent = parent.bro_up
            if (parent.bro_up is None) and (parent.proba < self.THRESHOLD):
                mid = up
                mid.proba += parent.proba * 1
                parent.Singleton(mid)
            else:
                down = mid
                mid = up
                up = Node(mid.S * self.alpha)
                down.proba += parent.proba * self.proba_down
                mid.proba += parent.proba * self.proba_mid
                up.proba = parent.proba * self.proba_up
                parent.Triplet(mid, up, down)

    def build_up_div(self, trunc_parent: Node, trunc_mid: Node, trunc_up: Node, trunc_down: Node, step: int) -> None:
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent
        while parent.bro_up is not None: # 
            parent = parent.bro_up
            fwd = self.forward(parent, step)
            if fwd < (up.S + mid.S) / 2:
                pass
            elif fwd > (up.S * self.alpha + up.S) / 2: 
                down = up
                mid = Node(down.S * self.alpha)
                up = Node(mid.S * self.alpha)
            else:
                down = mid
                mid = up
                up = Node(mid.S * self.alpha)
            parent.TripletDiv(mid, up, down)
            self._assign_dividend_probabilities(parent, mid, up, down, step)

    def build_down(self, trunc_parent: Node, trunc_mid: Node, trunc_up: Node, trunc_down: Node) -> None:
        # Reset parameters for the next "while loop"
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        # Create triplets while parents have brothers down
        while parent.bro_down is not None:
            parent = parent.bro_down
            # Test if parents is last lower parent of the column AND if its proba < THRESHOLD (for pruning)
            if (parent.bro_down is None) and (parent.proba < self.THRESHOLD):
                mid = down
                mid.proba += parent.proba * 1
                parent.Singleton(mid)
            # Else, no pruning
            else:
                up = mid
                mid = down
                down = Node(mid.S / self.alpha)
                up.proba += parent.proba * self.proba_up
                mid.proba += parent.proba * self.proba_mid
                down.proba = parent.proba * self.proba_down
                parent.Triplet(mid, up, down)
    # Build the lower Nodes of the column
    def build_down_div(self, trunc_parent: Node, trunc_mid: Node, trunc_up: Node, trunc_down: Node, step: int) -> None:
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        while parent.bro_down is not None:
            parent = parent.bro_down
            fwd = self.forward(parent, step)

            if fwd > (mid.S + down.S) / 2:
                pass
            elif fwd < (down.S / self.alpha + down.S) / 2: #
                up = down
                mid = Node(up.S / self.alpha)
                down = Node(mid.S / self.alpha)
            else:
                up = mid
                mid = down
                down = Node(mid.S / self.alpha)

            parent.TripletDiv(mid, up, down)
            self._assign_dividend_probabilities(parent, mid, up, down, step)

    def _assign_dividend_probabilities(
        self, parent: Node, mid: Node, up: Node, down: Node, step: int
) -> None:
        """ Assigne les probabilités (proba_up, proba_mid, proba_down) en date de dividende. """
        proba_up, proba_mid, proba_down = self.get_proba(parent, step)
        up.proba += parent.proba * proba_up
        try:
            mid.proba += parent.proba * proba_mid
        except:
            mid.proba = parent.proba * proba_mid
        try:
            down.proba += parent.proba * proba_down
        except:
            down.proba = parent.proba * proba_down

    def build_column(self, trunc_parent: Node, step: int) -> None: # Build the column of the tree
        trunc_mid = Node(self.forward(trunc_parent, step))
        trunc_mid.parent = trunc_parent
        trunc_up = Node(trunc_mid.S * self.alpha)
        trunc_down = Node(trunc_mid.S / self.alpha)
        trunc_mid.proba = trunc_parent.proba * self.proba_mid
        trunc_up.proba = trunc_parent.proba * self.proba_up
        trunc_down.proba = trunc_parent.proba * self.proba_down
        trunc_parent.Triplet(trunc_mid, trunc_up, trunc_down)

        if self.is_div_date(step):
            # Build the upper Nodes of the column
            self.build_up_div(trunc_parent, trunc_mid, trunc_up, trunc_down, step)
            # Build the upper Nodes of the column
            self.build_down_div(trunc_parent, trunc_mid, trunc_up, trunc_down, step)
        else:
            # Build the upper Nodes of the column
            self.build_up(trunc_parent, trunc_mid, trunc_up, trunc_down)
            # Build the upper Nodes of the column
            self.build_down(trunc_parent, trunc_mid, trunc_up, trunc_down)

    def build_tree(self) -> Node:
        root = self.root
        root_up = Node(self.root.S * self.alpha)
        root_down = Node(self.root.S / self.alpha)
        self.root.bro_up = root_up
        self.root.bro_down = root_down
        root_up.proba = 1/6
        root_down.proba = 1/6
        trunc_parent = root
        for step in range(self.n_steps):
            self.build_column(trunc_parent, step)
            trunc_parent = trunc_parent.child_mid
        return root

    def get_trunc_node(self, step: int) -> Node: # Get the t-th Node on the trunc
        if step > self.n_steps or step < 0:
            raise ValueError("0 <= step <= n_steps est obligatoire")
        trunc_node = self.root
        for _ in range(step):
            trunc_node = trunc_node.child_mid
        return trunc_node

    def get_node(self, step: int, height: int) -> Node:
        trunc_node = self.get_trunc_node(step)
        current_node = trunc_node
        if height >= 0:
            for _ in range(height):
                current_node = current_node.bro_up
        else:
            for _ in range(-height):
                current_node = current_node.bro_down
        return current_node

    # For pricing backward (we suppose that children's NFV is already computed)
    def average_child_value_no_div(self, current_node: Node) -> float:
        try:
            average = current_node.child_up.NFV * self.proba_up + current_node.child_mid.NFV * \
                      self.proba_mid + current_node.child_down.NFV * self.proba_down
        except:
            average = current_node.child_mid.NFV * 1

        return average

    def average_child_value_div(self, current_node: Node, step: int) -> float:

        proba_up, proba_mid, proba_down = self.get_proba(current_node, step)

        average = current_node.child_up.NFV * proba_up + current_node.child_mid.NFV * \
                  proba_mid + current_node.child_down.NFV * proba_down
        return average

    def average_child_value(self, current_node: Node, step: int) -> float:
        if self.is_div_date(step):
            average = self.average_child_value_div(current_node, step)
        else:
            average = self.average_child_value_no_div(current_node)

        # According to exec_type (European or American)
        if self.option.exercise == "european":
            return average
        elif self.option.exercise == "american":
            return max(average, self.option.payoff(current_node.S))
        else:
            raise ValueError("Execution type wrongly specified. Please only use 'European' or 'American'")

    def price(self, **kwargs) -> float:
        global trunc_node

        if self.tree_price is None:
            trunc_node = self.get_trunc_node(self.n_steps)
            self._initialize_terminal_nodes(trunc_node)

            step = self.n_steps
            while trunc_node.parent is not None:
                trunc_node = trunc_node.parent
                step -= 1
                self._propagate_column_backward(trunc_node, step)

            self.tree_price = self.root.NFV

        if kwargs.get("up"):
            return self.root.bro_up.NFV
        elif kwargs.get("down"):
            return self.root.bro_down.NFV
        else:
            return self.tree_price

    def _initialize_terminal_nodes(self, trunc_node: Node) -> None:
        """ Initialise les NFV de la colonne terminale (maturité). """
        trunc_node.NFV = self.option.payoff(trunc_node.S)
        up_node = trunc_node
        while up_node.bro_up is not None:
            up_node = up_node.bro_up
            up_node.NFV = self.option.payoff(up_node.S)
        down_node = trunc_node
        while down_node.bro_down is not None:
            down_node = down_node.bro_down
            down_node.NFV = self.option.payoff(down_node.S)

    def _propagate_column_backward(self, trunc_node: Node, step: int) -> None:
        """ Calcule les NFV d’une colonne à partir de la suivante. """
        trunc_node.NFV = self.df * self.average_child_value(trunc_node, step)
        up_node = trunc_node
        while up_node.bro_up is not None:
            up_node = up_node.bro_up
            up_node.NFV = self.df * self.average_child_value(up_node, step)
        down_node = trunc_node
        while down_node.bro_down is not None:
            down_node = down_node.bro_down
            down_node.NFV = self.df * self.average_child_value(down_node, step)


    def gap(self) -> float: # Calculate the gap 
        return (3 * self.market.S0 * (np.exp(self.market.sigma ** 2 * self.dt) - 1) * np.exp(
            2 * self.market.r * self.dt)) \
            / (8 * np.sqrt(2 * np.pi) * np.sqrt(np.exp(self.market.sigma ** 2 * self.T) - 1))

    def proba_check(self) -> List[float]:
        L: List[float] = []
        root = self.root
        L.append(root.proba)

        while root.child_mid is not None:
            root = root.child_mid
            trunc = root

            if root.proba < 0:
                raise ValueError("Proba négative")

            proba = root.proba
            proba += self._sum_branch_probas(trunc, direction="up")
            proba += self._sum_branch_probas(trunc, direction="down")

            L.append(round(proba, 12))

        print("Check OK")
        return L

    def _sum_branch_probas(self, start_node: Node, direction: str) -> float:
        """Additionne les probabilités dans la branche haut ou bas à partir d’un nœud donné."""
        if direction == "up":
            current = start_node.bro_up
            attr = "bro_up"
        elif direction == "down":
            current = start_node.bro_down
            attr = "bro_down"
        else:
            raise ValueError("direction must be 'up' or 'down'")

        total = 0.0
        while current is not None:
            if current.proba < 0:
                raise ValueError("Proba négative")
            total += current.proba
            current = getattr(current, attr)
        return total
