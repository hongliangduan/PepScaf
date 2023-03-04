from abc import ABC, abstractmethod


class ScafState(ABC):
    @abstractmethod
    def result(self):
        """
        this property should return:

         1 if peptide active
        -1 if peptide negative
         0 if there is a peptide
         None if result is unknown

        Returns
        -------
        int

        """
        pass

    def is_scaffold_over(self):
        """
        boolean indicating if the game is over,
        simplest implementation may just be
        `return self.get_result() is not None`

        Returns
        -------
        boolean

        """
        pass

    @abstractmethod
    def move(self, action):
        """
        consumes action and returns resulting PepGenState

        Parameters
        ----------
        action: AbstractGenAction

        Returns
        -------
        PepGenState

        """
        pass

    @abstractmethod
    def get_actions(self):
        """
        returns list of legal action at current peptide state
        Returns
        -------
        list of AbstractGenAction

        """
        pass


class AbstractScafAction(ABC):
    pass
