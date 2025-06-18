from typing import Literal, List, Dict
from tokenizers import Tokenizer, decoders, processors

'''
Specification:
You will be splitting each text transcript into tokenized text, which will then serve as inputs to your model. 
Additionally, you will need a method to convert the tokenized text back to its original form during inference or generation. 
Character-level tokenization is straightforward, while subword tokenization involves splitting a word into subwords rather than characters. 
The specific algorithm used for subword tokenization is [Byte Pair Encoding (BPE)](https://arxiv.org/pdf/1508.07909). 
Part of the assignment will involve exploring these different tokenization strategies to assess their impact on your model's performance.

The H4Tokenizer class provides tokenization functionality for the homework. 


1. Tokenizer Types:
   - Supports multiple tokenization strategies:
     * char: Character-level tokenization
     * 1k: Subword tokenization with 1000 vocab size
     * 5k: Subword tokenization with 5000 vocab size
     * 10k: Subword tokenization with 10000 vocab size

2. Special Tokens:
   - [PAD]: Used for padding sequences to same length
   - [UNK]: Represents unknown/out-of-vocabulary tokens
   - [MASK]: Used for masked language modeling
   - [SOS]: Marks start of sequence
   - [EOS]: Marks end of sequence
   - [BLANK]: Used for blank tokens in ASR

3. Key Operations:
   - encode(): Converts text to token IDs
   - decode(): Converts token IDs back to text
   - tokenize(): Splits text into tokens without converting to IDs
   - get_avg_chars_per_token(): Calculates character/token ratio for perplexity

4. Validation:
   - Automatically validates tokenizer at initialization
   - Shows example tokenization/detokenization
   - Displays configuration and special token IDs

Usage:
- Used by LMDataset and ASRDataset to process text data
- Ensures consistent tokenization across training/validation
- Handles special token addition and sequence conversion
'''

class H4Tokenizer:
    """
    A tokenizer class that supports character-level and subword tokenization strategies.
    Loads pre-trained tokenizers and provides methods for encoding/decoding text.

    Attributes:
        token_type (str): Type of tokenizer ('char', '1k', '5k', '10k')
        tokenizer (Tokenizer): The underlying tokenizer object
        pad_id (int): ID for padding token '[PAD]'
        unk_id (int): ID for unknown token '[UNK]'
        mask_id (int): ID for mask token '[MASK]'
        sos_id (int): ID for start-of-sequence token '[SOS]'
        eos_id (int): ID for end-of-sequence token '[EOS]'
        blank_id (int): ID for blank token '[BLANK]'
    """

    VALID_TYPES = ['char', '1k', '5k', '10k']

    def __init__(self, token_map: Dict[str, str], token_type: Literal['char', '1k', '5k', '10k']='char', validate: bool=True):
        """
        Initialize tokenizer from pre-trained file.

        Args:
            token_map: Maps token types to tokenizer file paths
            token_type: Type of tokenizer to load
            validate: Whether to validate the tokenizer
        
        Raises:
            ValueError: If invalid token_type provided
        """
        if token_type not in self.VALID_TYPES:
            raise ValueError(f"token_type must be one of {self.VALID_TYPES}")

        self.token_type = token_type
        self.tokenizer = Tokenizer.from_file(token_map[token_type])

        # Configure decoder based on tokenizer type
        if token_type != 'char':
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
            self.tokenizer.decoder = decoders.ByteLevel()
        else:
            self.tokenizer.decoder = decoders.Fuse()

        # Get vocabulary size
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Load special token IDs
        self.pad_id   = self.tokenizer.token_to_id("[PAD]")
        self.unk_id   = self.tokenizer.token_to_id("[UNK]") 
        self.mask_id  = self.tokenizer.token_to_id("[MASK]")
        self.sos_id   = self.tokenizer.token_to_id("[SOS]")
        self.eos_id   = self.tokenizer.token_to_id("[EOS]")
        self.blank_id = self.tokenizer.token_to_id("[BLANK]")

        if validate:    
            self._validate_tokenizer()

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of token strings
        """
        return self.tokenizer.encode(text).tokens

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool=False) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _validate_tokenizer(self) -> None:
        """
        Validate tokenizer functionality by testing basic operations.
        Prints diagnostic information and test results.
        """
        test_text = "[SOS]HI DEEP LEARNERS[EOS]"
        tokens    = self.tokenize(test_text)
        ids       = self.encode(test_text) 
        decoded   = self.decode(ids)

        print("="*80)
        title = f"Tokenizer Configuration ({self.token_type})" 
        print(f"{title:^80}")
        print("-"*80)
        print(f"{'Vocabulary size:':<20} {self.tokenizer.get_vocab_size()}")
        print("\nSpecial Tokens:")
        print(f"{'PAD:':<12} {self.pad_id:>6}")
        print(f"{'UNK:':<12} {self.unk_id:>6}")
        print(f"{'MASK:':<12} {self.mask_id:>6}")
        print(f"{'SOS:':<12} {self.sos_id:>6}")
        print(f"{'EOS:':<12} {self.eos_id:>6}")
        print(f"{'BLANK:':<12} {self.blank_id:>6}")
        print("\nValidation Example:")
        print("-"*80)
        print(f"{'Input text:':<12} {test_text}")
        print(f"{'Tokens:':<12} {tokens}")
        print(f"{'Token IDs:':<12} {ids}")
        print(f"{'Decoded:':<12} {decoded}")
        print("="*80)

    def get_avg_chars_per_token(self, token_ids: List[int], skip_special_tokens: bool = True) -> float:
        """
        Calculate average number of characters per token for given token IDs.

        Args:
            token_ids: List of token IDs to analyze
            skip_special_tokens: Whether to exclude special tokens from calculation

        Returns:
            Average number of characters per token
        """
        decoded_text = self.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if skip_special_tokens:
            token_count = sum(1 for id in token_ids if id not in 
                             [self.pad_id, self.unk_id, self.mask_id, 
                              self.sos_id, self.eos_id, self.blank_id])
        else:
            token_count = len(token_ids)
        
        return len(decoded_text) / token_count if token_count > 0 else 0