from collections import Counter, OrderedDict

import numpy as np
import pycld2 as cld2
import spacy
# from nltk.corpus import stopwords
from sklearn import preprocessing

ISO_STANDARDS = {"639_1": 0, "639_2/t": 1, "639_2/b": 2}
ISO_LANGUAGES = {"abkhaz": ['ab', 'abk', 'abk'], "afar": ['aa', 'aar', 'aar'],
                 "afrikaans": ['af', 'afr', 'afr'], "akan": ['ak', 'aka', 'aka'],
                 "albanian": ['sq', 'sqi', 'alb'], "amharic": ['am', 'amh', 'amh'],
                 "arabic": ['ar', 'ara', 'ara'], "aragonese": ['an', 'arg', 'arg'],
                 "armenian": ['hy', 'hye', 'arm'], "assamese": ['as', 'asm', 'asm'],
                 "avaric": ['av', 'ava', 'ava'], "avestan": ['ae', 'ave', 'ave'],
                 "aymara": ['ay', 'aym', 'aym'], "azerbaijani": ['az', 'aze', 'aze'],
                 "bambara": ['bm', 'bam', 'bam'], "bashkir": ['ba', 'bak', 'bak'],
                 "basque": ['eu', 'eus', 'baq'], "belarusian": ['be', 'bel', 'bel'],
                 "bengali": ['bn', 'ben', 'ben'], "bihari": ['bh', 'bih', 'bih'],
                 "bislama": ['bi', 'bis', 'bis'], "bosnian": ['bs', 'bos', 'bos'],
                 "breton": ['br', 'bre', 'bre'], "bulgarian": ['bg', 'bul', 'bul'],
                 "burmese": ['my', 'mya', 'bur'], "catalan": ['ca', 'cat', 'cat'],
                 "chamorro": ['ch', 'cha', 'cha'], "chechen": ['ce', 'che', 'che'],
                 "chichewa": ['ny', 'nya', 'nya'], "chinese": ['zh', 'zho', 'chi'],
                 "chuvash": ['cv', 'chv', 'chv'], "cornish": ['kw', 'cor', 'cor'],
                 "corsican": ['co', 'cos', 'cos'], "cree": ['cr', 'cre', 'cre'],
                 "croatian": ['hr', 'hrv', 'hrv'], "czech": ['cs', 'ces', 'cze'],
                 "danish": ['da', 'dan', 'dan'], "divehi": ['dv', 'div', 'div'],
                 "dutch": ['nl', 'nld', 'dut'], "dzongkha": ['dz', 'dzo', 'dzo'],
                 "english": ['en', 'eng', 'eng'], "esperanto": ['eo', 'epo', 'epo'],
                 "estonian": ['et', 'est', 'est'], "ewe": ['ee', 'ewe', 'ewe'],
                 "faroese": ['fo', 'fao', 'fao'], "fijian": ['fj', 'fij', 'fij'],
                 "finnish": ['fi', 'fin', 'fin'], "french": ['fr', 'fra', 'fre'],
                 "fula": ['ff', 'ful', 'ful'], "galician": ['gl', 'glg', 'glg'],
                 "georgian": ['ka', 'kat', 'geo'], "german": ['de', 'deu', 'ger'],
                 "greek ": ['el', 'ell', 'gre'], "guaraní": ['gn', 'grn', 'grn'],
                 "gujarati": ['gu', 'guj', 'guj'], "haitian": ['ht', 'hat', 'hat'],
                 "hausa": ['ha', 'hau', 'hau'], "hebrew": ['he', 'heb', 'heb'],
                 "herero": ['hz', 'her', 'her'], "hindi": ['hi', 'hin', 'hin'],
                 "hiri motu": ['ho', 'hmo', 'hmo'], "hungarian": ['hu', 'hun', 'hun'],
                 "interlingua": ['ia', 'ina', 'ina'], "indonesian": ['id', 'ind', 'ind'],
                 "interlingue": ['ie', 'ile', 'ile'], "irish": ['ga', 'gle', 'gle'],
                 "igbo": ['ig', 'ibo', 'ibo'], "inupiaq": ['ik', 'ipk', 'ipk'],
                 "ido": ['io', 'ido', 'ido'], "icelandic": ['is', 'isl', 'ice'],
                 "italian": ['it', 'ita', 'ita'], "inuktitut": ['iu', 'iku', 'iku'],
                 "japanese": ['ja', 'jpn', 'jpn'], "javanese": ['jv', 'jav', 'jav'],
                 "kalaallisut": ['kl', 'kal', 'kal'], "kannada": ['kn', 'kan', 'kan'],
                 "kanuri": ['kr', 'kau', 'kau'], "kashmiri": ['ks', 'kas', 'kas'],
                 "kazakh": ['kk', 'kaz', 'kaz'], "khmer": ['km', 'khm', 'khm'],
                 "kikuyu": ['ki', 'kik', 'kik'], "kinyarwanda": ['rw', 'kin', 'kin'],
                 "kyrgyz": ['ky', 'kir', 'kir'], "komi": ['kv', 'kom', 'kom'],
                 "kongo": ['kg', 'kon', 'kon'], "korean": ['ko', 'kor', 'kor'],
                 "kurdish": ['ku', 'kur', 'kur'], "kwanyama": ['kj', 'kua', 'kua'],
                 "latin": ['la', 'lat', 'lat'], "luxembourgish": ['lb', 'ltz', 'ltz'],
                 "ganda": ['lg', 'lug', 'lug'], "limburgish": ['li', 'lim', 'lim'],
                 "lingala": ['ln', 'lin', 'lin'], "lao": ['lo', 'lao', 'lao'],
                 "lithuanian": ['lt', 'lit', 'lit'], "luba-katanga": ['lu', 'lub', 'lub'],
                 "latvian": ['lv', 'lav', 'lav'], "manx": ['gv', 'glv', 'glv'],
                 "macedonian": ['mk', 'mkd', 'mac'], "malagasy": ['mg', 'mlg', 'mlg'],
                 "malay": ['ms', 'msa', 'may'], "malayalam": ['ml', 'mal', 'mal'],
                 "maltese": ['mt', 'mlt', 'mlt'], "māori": ['mi', 'mri', 'mao'],
                 "marathi": ['mr', 'mar', 'mar'], "marshallese": ['mh', 'mah', 'mah'],
                 "mongolian": ['mn', 'mon', 'mon'], "nauruan": ['na', 'nau', 'nau'],
                 "navajo": ['nv', 'nav', 'nav'], "northern ndebele": ['nd', 'nde', 'nde'],
                 "nepali": ['ne', 'nep', 'nep'], "ndonga": ['ng', 'ndo', 'ndo'],
                 "norwegian bokmål": ['nb', 'nob', 'nob'],
                 "norwegian nynorsk": ['nn', 'nno', 'nno'],
                 "norwegian": ['no', 'nor', 'nor'], "nuosu": ['ii', 'iii', 'iii'],
                 "southern ndebele": ['nr', 'nbl', 'nbl'], "occitan": ['oc', 'oci', 'oci'],
                 "ojibwe": ['oj', 'oji', 'oji'], "old bulgarian": ['cu', 'chu', 'chu'],
                 "oromo": ['om', 'orm', 'orm'], "oriya": ['or', 'ori', 'ori'],
                 "ossetian": ['os', 'oss', 'oss'], "panjabi": ['pa', 'pan', 'pan'],
                 "pāli": ['pi', 'pli', 'pli'], "persian": ['fa', 'fas', 'per'],
                 "polish": ['pl', 'pol', 'pol'], "pashto": ['ps', 'pus', 'pus'],
                 "portuguese": ['pt', 'por', 'por'], "quechua": ['qu', 'que', 'que'],
                 "romansh": ['rm', 'roh', 'roh'], "kirundi": ['rn', 'run', 'run'],
                 "romanian": ['ro', 'ron', 'rum'], "russian": ['ru', 'rus', 'rus'],
                 "sanskrit": ['sa', 'san', 'san'], "sardinian": ['sc', 'srd', 'srd'],
                 "sindhi": ['sd', 'snd', 'snd'], "northern sami": ['se', 'sme', 'sme'],
                 "samoan": ['sm', 'smo', 'smo'], "sango": ['sg', 'sag', 'sag'],
                 "serbian": ['sr', 'srp', 'srp'], "gaelic": ['gd', 'gla', 'gla'],
                 "shona": ['sn', 'sna', 'sna'], "sinhala": ['si', 'sin', 'sin'],
                 "slovak": ['sk', 'slk', 'slo'], "slovene": ['sl', 'slv', 'slv'],
                 "somali": ['so', 'som', 'som'], "southern sotho": ['st', 'sot', 'sot'],
                 "spanish": ['es', 'spa', 'spa'], "sundanese": ['su', 'sun', 'sun'],
                 "swahili": ['sw', 'swa', 'swa'], "swati": ['ss', 'ssw', 'ssw'],
                 "swedish": ['sv', 'swe', 'swe'], "tamil": ['ta', 'tam', 'tam'],
                 "telugu": ['te', 'tel', 'tel'], "tajik": ['tg', 'tgk', 'tgk'],
                 "thai": ['th', 'tha', 'tha'], "tigrinya": ['ti', 'tir', 'tir'],
                 "tibetan": ['bo', 'bod', 'tib'], "turkmen": ['tk', 'tuk', 'tuk'],
                 "tagalog": ['tl', 'tgl', 'tgl'], "tswana": ['tn', 'tsn', 'tsn'],
                 "tonga": ['to', 'ton', 'ton'], "turkish": ['tr', 'tur', 'tur'],
                 "tsonga": ['ts', 'tso', 'tso'], "tatar": ['tt', 'tat', 'tat'],
                 "twi": ['tw', 'twi', 'twi'], "tahitian": ['ty', 'tah', 'tah'],
                 "uyghur": ['ug', 'uig', 'uig'], "ukrainian": ['uk', 'ukr', 'ukr'],
                 "urdu": ['ur', 'urd', 'urd'], "uzbek": ['uz', 'uzb', 'uzb'],
                 "venda": ['ve', 'ven', 'ven'], "vietnamese": ['vi', 'vie', 'vie'],
                 "volapük": ['vo', 'vol', 'vol'], "walloon": ['wa', 'wln', 'wln'],
                 "welsh": ['cy', 'cym', 'wel'], "wolof": ['wo', 'wol', 'wol'],
                 "western frisian": ['fy', 'fry', 'fry'], "xhosa": ['xh', 'xho', 'xho'],
                 "yiddish": ['yi', 'yid', 'yid'], "yoruba": ['yo', 'yor', 'yor'],
                 "zhuang  chuang": ['za', 'zha', 'zha'], "zulu": ['zu', 'zul', 'zul']}

SPACY_LANGUAGE_MAPPING = {
    # "spanish": "es_core_news_sm",
    "spanish": ["es_core_news_lg", "es_core_news_md", "es_core_news_sm"],
    # "english": "en_core_web_sm",
    "english": ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"],
    # "portuguese": "pt_core_news_sm",
    # "italian": "it_core_news_sm",
    # "french": "fr_core_news_sm",
    # "german": "de_core_news_sm"
}


def detect_language(text):
    try:
        # text = bytes(text, 'utf-8').decode('utf-8', 'backslashreplace')
        detected = cld2.detect(text)
        # Example of 'detected'
        # (True, 75266, (('SPANISH', 'es', 90, 507.0), ('ENGLISH', 'en', 9, 541.0), ('Unknown', 'un', 0, 0.0)))
        if detected[0]:
            lang = detected[2][0][0].lower()
            lang_code = detected[2][0][1]  # lang_code is always in ISO 639_1 standard
        else:
            lang = lang_code = None
    except Exception as err:
        raise Exception("TextProcessor::detect_language: " + str(err))
    return lang, lang_code


class DocumentContainer:
    def __init__(self, document, language=None, iso_standard="639_1"):
        self.nlp_doc = document
        self.language = language
        if not self.language:
            self.language = self._get_lang_from_lang_code(self.nlp_doc.lang)
        self.iso_standard = iso_standard

    def _get_lang_from_lang_code(self, lang_code):
        idx = ISO_STANDARDS[self.iso_standard]
        for key, value in ISO_LANGUAGES.items():
            if value[idx] == lang_code:
                return key
        raise ValueError("ISO language code {} not recognizable".format(lang_code))

    @staticmethod
    def _get_tkn_attrs(token, attrs_names):
        attr_tkn_list = []
        for name in attrs_names:
            if name == 'lemma_':
                attr_tkn_list.append(token.lemma_ if token.lemma_ != "-PRON-" else token.lower_)
            else:
                attr_tkn_list.append(getattr(token, name))
        return attr_tkn_list

    def get_tokens_property_list(self, prop_names, check_attributes=(), check_if_false=False):
        attr_list = []
        for tok in self.nlp_doc:
            valid_token = True
            if check_attributes:
                if any([getattr(tok, attr) for attr in check_attributes]):
                    valid_token = not check_if_false
                else:
                    valid_token = check_if_false
            if valid_token:
                attr_list.append(self._get_tkn_attrs(tok, prop_names))
                # attr_list.append([getattr(tok, name) for name in prop_names])
        return attr_list

    def get_tokens(self, lowercase=True):
        return self.get_tokens_property_list(["lower_"] if lowercase else ["text"])

    def get_lemmas(self):
        return self.get_tokens_property_list(["lemma_"])

    def get_numerical_tokens(self):
        return self.get_tokens_property_list(["text"], check_attributes=("is_digit",))

    def get_alpha_tokens(self, lowercase=True):
        return self.get_tokens_property_list(["lower_"] if lowercase else ["text"], check_attributes=("is_alpha",))

    def get_urllike_tokens(self):
        return self.get_tokens_property_list(["text"], check_attributes=("like_url",))

    def get_emaillike_tokens(self):
        return self.get_tokens_property_list(["text"], check_attributes=("like_email",))

    def get_filtered_dimensions(self, properties=None, filter_include=(), filter_exclude=(), flatten=False):
        attributes = {"is_stop", "is_punct", "is_quote", "is_currency", "is_space", "is_digit"}
        attributes = attributes.union(filter_include)
        attributes = list(attributes.difference(filter_exclude))
        # convert 'properties' in a list of attributes
        if properties is None:
            properties = ['lemma_']
        elif isinstance(properties, str):
            properties = [properties]
        elif isinstance(properties, tuple):
            properties = list(properties)
        if len(properties) == 0:
            raise ValueError("Should be at least one property to return")
        if len(properties) > 1 and flatten:
            raise AttributeError("'flatten' property may be True only when there is one single property to return")

        # always add 'text' if verifying stop words
        if "is_stop" in attributes:
            properties.append('text')
        features = self.get_tokens_property_list(properties, check_attributes=attributes, check_if_false=True)
        # Convert all features to lowercase, because spacy let capitalized words stays intact.
        # Because nltk stopwords definition is more complete than spacy, here we check this.
        # WE SHOULDN'T INCLUDE THE WHOLE NLTK CORPUS JUST TO USE THE STOPWORDS
        # if "is_stop" in attributes:
        #     features = [f[:-1] for f in features if f[-1] not in stopwords.words(self.language)]

        if flatten:
            features = [f[0] for f in features]

        # if self.lang == "spanish":
        # features = [feat for feat in features if len(feat) > 1]
        return features

    def get_bag_of_words(self, order='numerical', **to_filter_kwargs):
        if 'properties' in to_filter_kwargs and not isinstance(to_filter_kwargs['properties'], str) and len(
                to_filter_kwargs['properties']) > 1:
            raise ValueError('Must return only one token property')
        to_filter_kwargs['flatten'] = True
        cleaned_features = self.get_filtered_dimensions(**to_filter_kwargs)
        # tf = {}
        # for feature in cleaned_features:
        #     tf[feature] = tf.get(feature, 0) + 1
        tf = Counter(cleaned_features)
        if order != 'numerical':
            return self._order_dict(tf, order_method=order)
        return tf

    @staticmethod
    def _order_dict(dict_, order_method='alphabetical'):
        allowed_methods = ("alphabetical", "numerical")
        if order_method not in allowed_methods:
            all_in_or = " or ".join(allowed_methods)
            raise ValueError("Wrong value of norm, {} expected".format(all_in_or))
        ordered_dict = OrderedDict()
        if order_method == allowed_methods[0]:
            ordered_keys = sorted(dict_)
        else:
            ordered_keys = sorted(dict_, key=dict_.get, reverse=True)
        for key in ordered_keys:
            ordered_dict[key] = dict_[key]
        return ordered_dict

    def get_unit_vector(self, norm='l2', order='numerical', **token_kwargs):
        if norm not in ('l1', 'l2'):
            raise ValueError("Wrong value of norm, 'l1' or 'l2' expected")
        bog = self.get_bag_of_words(order=order, **token_kwargs)
        word_list = []
        values_list = []
        if type(bog) == Counter:
            for word, frequency in bog.most_common():
                word_list.append(word)
                values_list.append(frequency)
        else:
            for word, frequency in bog.items():
                word_list.append(word)
                values_list.append(frequency)
        x = np.asarray([values_list], dtype=np.float)
        # X.shape is a tuple of shape (num_elements_in_array, num_subelements_inside_each_element)
        features_length = x.shape[1]
        x_normalized = preprocessing.normalize(x, norm=norm)
        full = {word_list[i]: x_normalized[0, i] for i in range(features_length)}
        return full


class LanguageProcessor:
    def __init__(self, language=None, language_code=None, iso_standard="639_1", **nlp_kwargs):
        if language is None and language_code is None:
            raise ValueError("Must specify either language or language_code")
        if ISO_STANDARDS.get(iso_standard) is None:
            raise ValueError("ISO standard {} not recognizable".format(iso_standard))
        if language is not None and language not in ISO_LANGUAGES:
            raise ValueError("Language {} not recognizable".format(language))

        self.iso_standard = iso_standard
        self.lang = language
        self.lang_iso_code = language_code
        if language is None:
            self.lang = self._get_lang_from_lang_code(language_code)
        elif language_code is None:
            self.lang_iso_code = self._get_lang_code_from_lang(language)

        # disable_pipes = ("ner", "textcat", "entity_ruler", "merge_entities") if lite_nlp else ()
        self.nlp = self.make_nlp(**nlp_kwargs)

    def __call__(self, text, disable_pipes=None, enable_pipes=None):
        """

        :param text: Text to process
        :param disable_pipes: The name(s) of the pipes to disable
        :param enable_pipes: The name(s) of the pipes to enable - all others will be disabled
        :return: DocumentContainer
        """
        if disable_pipes or enable_pipes:
            with self.nlp.select_pipes(disable=disable_pipes, enable=enable_pipes):
                document = self.nlp(text)
                return DocumentContainer(document, self.lang, self.iso_standard)
        document = self.nlp(text)
        return DocumentContainer(document, self.lang, self.iso_standard)

    def _get_lang_from_lang_code(self, lang_code):
        idx = ISO_STANDARDS[self.iso_standard]
        for key, value in ISO_LANGUAGES.items():
            if value[idx] == lang_code:
                return key
        raise ValueError("ISO language code {} not recognizable".format(lang_code))

    def _get_lang_code_from_lang(self, language_name):
        idx = ISO_STANDARDS[self.iso_standard]
        return ISO_LANGUAGES[language_name][idx]

    def make_nlp(self, add_pipes=(), remove_pipes=(), light_model=True):
        lang_to_load = SPACY_LANGUAGE_MAPPING[self.lang]
        nlp = None
        if light_model:
            lang_to_load = lang_to_load[::-1]
        for model_name in lang_to_load:
            try:
                nlp = spacy.load(model_name)
                break
            except IOError:
                continue
        if nlp is None:
            raise IOError(
                "No suitable spacy model was found for language {}, please visit https://spacy.io/models".format(
                    self.lang))
        if add_pipes is not None and len(add_pipes) > 0:
            for pipe in add_pipes:
                # each pipe should be a dict with add_pipe arguments
                try:
                    nlp.add_pipe(**pipe)
                except:
                    continue
        if remove_pipes is not None and len(remove_pipes) > 0:
            for pipe in remove_pipes:
                nlp.remove_pipe(pipe)
        return nlp
