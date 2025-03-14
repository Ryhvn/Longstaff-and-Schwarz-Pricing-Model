import numpy as np
from Node import Node
from PricingModel import Model  # Importation de la classe mère

class TreeModel(Model):
    def __init__(self, market, option, pricing_date, n_paths, n_steps, THRESHOLD=1e-7):
        super().__init__(market, option, pricing_date, n_paths, n_steps)  # Appel du constructeur parent

        self.THRESHOLD = THRESHOLD
        self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.dt))
        self.proba_down = (np.exp(self.market.sigma ** 2 * self.dt) - 1) \
                          / ((1 - self.alpha) * (self.alpha ** (-2) - 1))
        self.proba_up = self.proba_down / self.alpha
        self.proba_mid = 1 - self.proba_down - self.proba_up
        self.root = Node(self.market.S0)
        self.root.proba = 1
        self.BuildTree()

    def IsDivDate(self, step):
        return step < self.t_div <= (step + 1) if self.t_div is not None else False

    def Forward(self, parent, step):
        if self.IsDivDate(step):
            return parent.S * np.exp(self.market.r * self.dt) - self.market.dividend
        return parent.S * np.exp(self.market.r * self.dt)

    def GetProba(self, div_node, step):
        average = self.Forward(div_node, step)
        var = div_node.S ** 2 * np.exp(2 * self.market.r * self.dt) * (np.exp(self.market.sigma ** 2 * self.dt) - 1)

        proba_down = (div_node.child_mid.S ** (-2) * (var + average ** 2) - 1 - (self.alpha + 1) *
                      (div_node.child_mid.S ** (-1) * average - 1)) / ((1 - self.alpha) * (self.alpha ** (-2) - 1))
        proba_up = (div_node.child_mid.S ** (-1) * average - 1 - (self.alpha ** (-1) - 1) *
                    proba_down) / (self.alpha - 1)
        proba_mid = 1 - proba_down - proba_up
        return proba_up, proba_mid, proba_down

    def BuildUp(self, trunc_parent, trunc_mid, trunc_up, trunc_down):
        # Set parameters for the "while loop"
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        # Create triplets while parents have brothers up
        while parent.bro_up is not None:
            parent = parent.bro_up
            # Test if parents is last upper parent of the column AND if its proba < THRESHOLD (for pruning)
            if (parent.bro_up is None) and (parent.proba < self.THRESHOLD):
                mid = up
                mid.proba += parent.proba * 1
                parent.Singleton(mid)
            # Else, no pruning
            else:
                down = mid
                mid = up
                up = Node(mid.S * self.alpha)
                down.proba += parent.proba * self.proba_down
                mid.proba += parent.proba * self.proba_mid
                up.proba = parent.proba * self.proba_up
                parent.Triplet(mid, up, down)

    def BuildUpDiv(self, trunc_parent, trunc_mid, trunc_up, trunc_down, step):
        # Similar to BuildUp, used for Dividend dates. We suppose no pruning on this column
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        while parent.bro_up is not None:
            parent = parent.bro_up
            if self.Forward(parent, step) < (up.S + mid.S) / 2:
                pass

            elif self.Forward(parent, step) > (up.S * self.alpha + up.S) / 2:
                down = up
                mid = Node(down.S * self.alpha)
                up = Node(mid.S * self.alpha)

            else:
                down = mid
                mid = up
                up = Node(mid.S * self.alpha)

            parent.TripletDiv(mid, up, down)
            proba_up, proba_mid, proba_down = self.GetProba(parent, step)
            try:
                up.proba += parent.proba * proba_up
            except:
                up.proba = parent.proba * proba_up
            try:
                mid.proba += parent.proba * proba_mid
            except:
                mid.proba = parent.proba * proba_mid
            down.proba += parent.proba * proba_down

    def BuildDown(self, trunc_parent, trunc_mid, trunc_up, trunc_down):
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

    def BuildDownDiv(self, trunc_parent, trunc_mid, trunc_up, trunc_down, step):
        # Similar to BuildDown, used for Dividend dates. We suppose no pruning on this column
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        while parent.bro_down is not None:
            parent = parent.bro_down
            if self.Forward(parent, step) > (mid.S + down.S) / 2:
                pass

            elif self.Forward(parent, step) < (down.S / self.alpha + down.S) / 2:
                up = down
                mid = Node(up.S / self.alpha)
                down = Node(mid.S / self.alpha)

            else:
                up = mid
                mid = down
                down = Node(mid.S / self.alpha)

            parent.TripletDiv(mid, up, down)
            proba_up, proba_mid, proba_down = self.GetProba(parent, step)
            up.proba += parent.proba * proba_up
            try:
                mid.proba += parent.proba * proba_mid
            except:
                mid.proba = parent.proba * proba_mid
            try:
                down.proba += parent.proba * proba_down
            except:
                down.proba = parent.proba * proba_down

    def BuildColumn(self, trunc_parent, step):
        trunc_mid = Node(self.Forward(trunc_parent, step))
        trunc_mid.parent = trunc_parent
        trunc_up = Node(trunc_mid.S * self.alpha)
        trunc_down = Node(trunc_mid.S / self.alpha)
        trunc_mid.proba = trunc_parent.proba * self.proba_mid
        trunc_up.proba = trunc_parent.proba * self.proba_up
        trunc_down.proba = trunc_parent.proba * self.proba_down
        trunc_parent.Triplet(trunc_mid, trunc_up, trunc_down)

        if self.IsDivDate(step):
            self.BuildUpDiv(trunc_parent, trunc_mid, trunc_up, trunc_down, step)
            self.BuildDownDiv(trunc_parent, trunc_mid, trunc_up, trunc_down, step)
        else:
            self.BuildUp(trunc_parent, trunc_mid, trunc_up, trunc_down)
            self.BuildDown(trunc_parent, trunc_mid, trunc_up, trunc_down)

    def BuildTree(self):
        root = self.root
        trunc_parent = root
        for step in range(self.n_steps):
            self.BuildColumn(trunc_parent, step)
            trunc_parent = trunc_parent.child_mid
        return root

    def GetTruncNode(self, step):
        if step > self.n_steps or step < 0:
            raise ValueError("0 <= step <= n_steps est obligatoire")
        trunc_node = self.root
        for _ in range(step):
            trunc_node = trunc_node.child_mid
        return trunc_node

    def GetNode(self, step, height):
        trunc_node = self.GetTruncNode(step)
        current_node = trunc_node
        if height >= 0:
            for _ in range(height):
                current_node = current_node.bro_up
        else:
            for _ in range(-height):
                current_node = current_node.bro_down
        return current_node

    # For pricing backward (we suppose that children's NFV is already computed)
    def AverageChildValueNoDiv(self, current_node):
        try:
            average = current_node.child_up.NFV * self.proba_up + current_node.child_mid.NFV * \
                      self.proba_mid + current_node.child_down.NFV * self.proba_down
        except:
            average = current_node.child_mid.NFV * 1

        return average

    def AverageChildValueDiv(self, current_node, step):

        proba_up, proba_mid, proba_down = self.GetProba(current_node, step)

        average = current_node.child_up.NFV * proba_up + current_node.child_mid.NFV * \
                  proba_mid + current_node.child_down.NFV * proba_down
        return average

    def AverageChildValue(self, current_node, step):
        if self.IsDivDate(step):
            average = self.AverageChildValueDiv(current_node, step)
        else:
            average = self.AverageChildValueNoDiv(current_node)

        # According to exec_type (European or American)
        if self.option.exercise == "european":
            return average
        elif self.option.exercise == "american":
            return max(average, self.option.payoff(current_node.S))
        else:
            raise ValueError("Execution type wrongly specified. Please only use 'European' or 'American'")

    def price(self):
        trunc_node = self.GetTruncNode(self.n_steps)
        trunc_node.NFV = self.option.payoff(trunc_node.S)

        up_node, down_node = trunc_node, trunc_node
        while up_node.bro_up is not None:
            up_node = up_node.bro_up
            up_node.NFV = self.option.payoff(up_node.S)
        while down_node.bro_down is not None:
            down_node = down_node.bro_down
            down_node.NFV = self.option.payoff(down_node.S)

        step = self.n_steps
        while trunc_node.parent is not None:
            trunc_node = trunc_node.parent
            trunc_node.NFV = self.df * self.AverageChildValue(trunc_node, step - 1)

            up_node, down_node = trunc_node, trunc_node
            while up_node.bro_up is not None:
                up_node = up_node.bro_up
                up_node.NFV = self.df * self.AverageChildValue(up_node, step - 1)
            while down_node.bro_down is not None:
                down_node = down_node.bro_down
                down_node.NFV = self.df * self.AverageChildValue(down_node, step - 1)

            step -= 1

        return trunc_node.NFV

    def Gap(self):
        return (3 * self.market.S0 * (np.exp(self.market.sigma ** 2 * self.dt) - 1) * np.exp(
            2 * self.market.r * self.dt)) \
            / (8 * np.sqrt(2 * np.pi) * np.sqrt(np.exp(self.market.sigma ** 2 * self.T) - 1))

    def ProbaCheck(self):
        L = []
        root = self.root
        L.append(root.proba)

        while root.child_mid is not None:
            root = root.child_mid
            trunc = root
            if root.proba < 0:
                raise ValueError("Proba négative")

            proba = root.proba
            while root.bro_up is not None:
                root = root.bro_up
                if root.proba < 0:
                    raise ValueError("Proba négative")
                proba += root.proba

            root = trunc
            while root.bro_down is not None:
                root = root.bro_down
                if root.proba < 0:
                    raise ValueError("Proba négative")
                proba += root.proba

            root = trunc
            L.append(round(proba, 12))

        print("Check OK")
        return L
