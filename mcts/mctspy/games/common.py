from abc import ABC, abstractmethod


class PepState(ABC):
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

    def is_peptide_over(self):
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


class AbstractGenAction(ABC):
    pass


class TwoPlayersAbstractGameState(ABC):
    @abstractmethod
    def game_result(self):
        """
        this property should return:

         1 if player #1 wins
        -1 if player #2 wins
         0 if there is a draw
         None if result is unknown

        Returns
        -------
        int

        """
        pass

    @abstractmethod
    def is_game_over(self):
        """
        boolean indicating if the game is over,
        simplest implementation may just be
        `return self.game_result() is not None`

        Returns
        -------
        boolean

        """
        pass

    @abstractmethod
    def move(self, action):
        """
        consumes action and returns resulting TwoPlayersAbstractGameState

        Parameters
        ----------
        action: AbstractGameAction

        Returns
        -------
        TwoPlayersAbstractGameState

        """
        pass

    @abstractmethod
    def get_legal_actions(self):
        """
        returns list of legal action at current game state
        Returns
        -------
        list of AbstractGameAction

        """
        pass


class AbstractGameAction(ABC):
    pass
