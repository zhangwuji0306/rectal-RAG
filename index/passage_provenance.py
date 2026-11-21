"""Replay passage-v2 with original-codepoint provenance; no I/O or model loading.

The supplied tokenizer and reference chunker must be the indexed versions.
All replayed text/sections must equal the reference chunker before returning any
mapping. Token-decoded windows without an unambiguous exact local match remain
unbound. Synthetic whitespace is explicit, never an original-source offset.
"""
import re


class _Text:
    def __init__(self, text, positions):
        self.text, self.positions = text, positions

    def part(self, start, end):
        return _Text(self.text[start:end], self.positions[start:end])

    def strip(self):
        start = len(self.text) - len(self.text.lstrip())
        end = len(self.text.rstrip())
        return self.part(start, max(start, end))


def _token_offsets(text, tokenizer, ids):
    if not callable(tokenizer):
        return None
    try:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except NotImplementedError:
        return None
    offsets = encoding.get('offset_mapping')
    if encoding.get('input_ids') != ids or offsets is None or len(offsets) != len(ids):
        return None
    previous_start = previous_end = 0
    for offset in offsets:
        if not isinstance(offset, (tuple, list)) or len(offset) != 2:
            return None
        start, end = offset
        if (type(start) is not int or type(end) is not int
                or not previous_start <= start < end <= len(text)
                or end < previous_end):
            return None
        previous_start, previous_end = start, end
    return offsets


def replay(original, tokenizer, reference_chunker, *, max_tokens=800, overlap=80,
           min_tokens=16):
    """Return complete chunks plus ordered source segments, or explicit unbound.

    Repeated passages are resolved by deterministic chunk-index replay, never by
    picking a global substring. Universal-newline conversion is tracked back to
    the unmodified UTF-8-decoded original. No offsets into normalized text escape.
    """
    chars, positions = [], []
    i = 0
    while i < len(original):
        if original[i:i+2] == '\r\n':
            chars.append('\n'); positions.append(i+1); i += 2
        else:
            chars.append('\n' if original[i] == '\r' else original[i])
            positions.append(i if original[i] != '\r' else None); i += 1
    normalized = _Text(''.join(chars), positions)

    def join(parts, separator):
        text, pos = [], []
        for index, part in enumerate(parts):
            if index:
                text.append(separator); pos.extend([None] * len(separator))
            text.append(part.text); pos.extend(part.positions)
        return _Text(''.join(text), pos)

    def split(value, pattern):
        parts, start = [], 0
        for match in re.finditer(pattern, value.text):
            parts.append(value.part(start, match.start())); start = match.end()
        parts.append(value.part(start, len(value.text)))
        return parts

    def count(value):
        return len(tokenizer.encode(value.text, add_special_tokens=False))

    def windows(value):
        ids = tokenizer.encode(value.text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return [value]
        if max_tokens <= overlap:
            raise ValueError('overlap must be smaller than max_tokens')
        offsets = _token_offsets(value.text, tokenizer, ids)
        out, start = [], 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            piece = tokenizer.decode(ids[start:end], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False).strip()
            if piece:
                at = -1
                if offsets is not None:
                    left, right = offsets[start][0], offsets[end - 1][1]
                    at = value.text.find(piece, left, right)
                    if at >= 0 and value.text.find(piece, at + 1, right) >= 0:
                        at = -1
                if at < 0:
                    out.append(_Text(piece, [None] * len(piece)))
                else:
                    out.append(value.part(at, at+len(piece)))
            if end == len(ids):
                break
            start += max_tokens - overlap
        return out

    def sentences(para):
        units = [u.strip() for u in split(para, r'(?<=[.!?])\s+|\n') if u.text.strip()]
        out, buf, bt = [], [], 0
        for u in units:
            ut = count(u)
            if ut > max_tokens:
                if buf:
                    out.append(join(buf, ' ')); buf, bt = [], 0
                out.extend(windows(u)); continue
            if buf and bt + ut > max_tokens:
                out.append(join(buf, ' ')); buf, bt = [], 0
            buf.append(u); bt += ut
        if buf:
            out.append(join(buf, ' '))
        return out

    chunks, heading, buf = [], [], []

    def emit(value, section):
        chunks.append((value, section))

    def flush():
        nonlocal buf
        if not buf:
            return
        body = join(buf, '\n').strip(); buf = []
        section = ' > '.join(t for _, t in heading)
        if not body.text or count(body) < min_tokens:
            return
        if count(body) <= max_tokens:
            emit(body, section); return
        paras = [p.strip() for p in split(body, r'\n\n') if p.text.strip()]
        acc, total = [], 0
        for p in paras:
            pt = count(p)
            if pt > max_tokens:
                if acc:
                    emit(join(acc, '\n\n'), section); acc, total = [], 0
                for sub in sentences(p):
                    emit(sub, section)
                continue
            if acc and total + pt > max_tokens:
                emit(join(acc, '\n\n'), section)
                carry, ct = [], 0
                for pp in reversed(acc):
                    t = count(pp)
                    if ct + t > overlap:
                        break
                    carry.insert(0, pp); ct += t
                acc, total = carry, ct
            acc.append(p); total += pt
        if acc:
            emit(join(acc, '\n\n'), section)

    fence = False
    # splitlines() semantics include more than LF; retain exact source positions.
    at = 0
    for line in normalized.text.splitlines(keepends=True):
        bare = line.splitlines()[0] if line.splitlines() else ''
        ln = normalized.part(at, at+len(bare)); at += len(line)
        stripped = ln.text.strip()
        if stripped.startswith('```'):
            fence = not fence; continue
        if fence or stripped.startswith('![') or stripped.startswith('<!--'):
            continue
        match = re.match(r'^(#{1,4})\s+(.*)$', ln.text)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            while heading and heading[-1][0] >= level:
                heading.pop()
            heading.append((level, title))
        else:
            buf.append(ln)
    flush()
    expected = reference_chunker(normalized.text, tokenizer, max_tokens=max_tokens, overlap=overlap)
    if expected != [dict(text=t.text, section=s) for t, s in chunks]:
        raise ValueError('provenance replay differs from indexed chunker')
    result = []
    for value, section in chunks:
        segments, cursor, last_end = [], 0, -1
        reason = None
        while cursor < len(value.text):
            start = value.positions[cursor]; end = cursor + 1
            if start is None:
                while end < len(value.text) and value.positions[end] is None:
                    end += 1
                literal = value.text[cursor:end]
                if not literal.isspace():
                    reason = 'token_decoding_not_exactly_locatable'; break
                segments.append(dict(separator=literal))
            else:
                while end < len(value.text) and value.positions[end] == start + end - cursor:
                    end += 1
                source_end = start + end - cursor
                if start < last_end:
                    reason = 'nonmonotonic_local_token_match'; break
                assert original[start:source_end] == value.text[cursor:end]
                segments.append(dict(start=start, end=source_end)); last_end = source_end
            cursor = end
        result.append(dict(text=value.text, section=section,
                           status='unbound' if reason else 'bound', reason=reason,
                           source_segments=None if reason else segments))
    return result
