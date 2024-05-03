/*
Copyright 2024 Datalust Pty. Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*!
A disk-friendly immutable hashmap for 32bit hashes of
keys to a set of 32bit page hits. What defines
a page is up to the caller; they may be indexes into a file
or some other structure. The choice of hashing function is also
up to the caller; it just needs to be stable.

It's logically like a `HashMap<u32, HashSet<u32>>`.
Maps are expensive to create, but are cheap to lookup.
The layout and implementation are optimized for the case
where a hash is not present (and so carries no hits).

# Input requirements

It must be valid for the map to return _more_ hits than
were originally inserted for a given key. This can happen
in case of hash collisions, or if a key carries more
hits than a saturation threshold.

# Format

```text
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─
│BO│BO│BO│PO│PO│PO│PO│HV│HV│HV│HV│PV│PV│PV│PV│PV│PV│..
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─
```

- **BO**: 32bit bucket offset
- **PO**: 32bit pages offset
- **HV**: 32bit hash value
- **PV**: 32bit page value

```text
  Header                     Bucket
┌────┴───┬──────────────────────┴──────────────────┐
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─
│BO│BO│BO│PO│PO│PO│PO│HV│HV│HV│HV│PV│PV│PV│PV│PV│PV│..
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─
         └─────┬─────┴──────┬────┴─────────┬───────┘
         │  Pages         Hash   │        Page     │
         │ offsets       values  │       values    │
         └───────────┬───────────┴─────────┬───────┘
                   Keys                  Values
```

The format is bucket-based. Each bucket is split into
two parts. All 32bit page offsets are packed together at
the start, followed by all 32bit hash values. You can think
of this section as storing the keys of the map. Next come
the page values. You can think of these as the values in
the map. Pages offsets and hash values are always the same
length and unique, but page values may be repeated.

# Finding hits

```text
                ┌─────4─────┐
┌──┬──┬──┬──┬──┐▼─┬──┬──┬──┬┴─┬──┬──┬──┬──┬──┬──┬──┬─
│BO│BO│BO│PO│PO│PO│PO│HV│HV│HV│HV│PV│PV│PV│PV│PV│PV│..
└▲─┴──┴──┘▲─┴──┴──┴──┘▲─┴──┘▲─┴──┴──┴──┘▲─┴▲─┘▲─┴──┴─
 1        2     │     │     │           │  │  │
 └────────┴───────────┴─3──►┘           5  5  5
                └───────────────────────┴──┴─►┘
```

1. Use the hash of the lookup value to find the bucket
   it belongs to. The length of the bucket is the data
   up to the next bucket's offset, or the end of the map
   if it's the last one.
2. Calculate the number of hashes in the bucket by looking
   at the first page offset, which will always point
   to the start of the page value set.
3. Scan through hash values until a match is found.
4. Offset back from the match's index to find its page
   value offset. The number of pages to read is the data
   up to the next hash's page offset, or the end of the
   bucket if it's the last one.
5. Return each page value as a hit.

# Trade-offs

This implementation is approximate; it only guarantees that
at least the set of pages inserted for a given value will
be returned. A value may return saturated or additional pages
in the case of hash collisions or hashes covering a lot of pages.
*/

use std::{
    cmp,
    collections::{BTreeMap, HashSet},
    io::{self, Write},
};

/*
A sample utility that creates a map if it doesn't already exist and reads hits from it.

Usage:

`diskmap 4`
*/
fn main() {
    // If the map already exists then use it
    let map = if let Ok(map) = std::fs::read("./diskmap.data") {
        map
    }
    // If the map doesn't exist then create it
    else {
        println!("creating map");

        // Our builder is for a file with 70 pages
        // The map contains 1000 hashes, each with up to 9 page hits
        let mut builder = DiskMap4Builder::new(70);
        for i in 0..1000 {
            for p in 0..(i % 10) {
                builder.mark_hit(i, p);
            }
        }

        // Write the map into a file
        let mut file = std::fs::File::create("./diskmap.data").unwrap();
        builder.write(&mut file).unwrap();
        drop(file);

        // Read the map from the file
        // Usually you'd mmap this, but for this sample we just buffer the whole lot
        std::fs::read("./diskmap.data").unwrap()
    };

    println!("reading map");

    let id = std::env::args().skip(1)
        .next()
        .ok_or("usage: `diskmap n`, where `n` is a value between 0..1000")
        .unwrap()
        .parse()
        .unwrap();

    // Reading maps is always infallible
    // If the map is invalid or corrupted we return a sentinal value
    // indicating all pages are hits. This is correct, but inefficient
    // for callers
    let map = DiskMap4::read(&map);

    println!("{:?}", map.hits(id));
}

/**
The number of bytes per bucket.

Larger values mean more scanning for slightly less space.
The optimal size here is one vector-width of hashes.
On ARM with Neon that's 4, and on x86 with AVX2 that's 8.
*/
#[cfg(target_arch = "aarch64")]
const TARGET_BUCKET_SIZE_HASH_VALUES: usize = 4;
#[cfg(not(target_arch = "aarch64"))]
const TARGET_BUCKET_SIZE_HASH_VALUES: usize = 8;

/**
If a hash covers more than `npages / SATURATED_THRESHOLD`
then store `SATURATED_PAGE` instead.

This value protects against degenerate inputs where a single
hash stores a very large number of pages, causing a single bucket
to grow unreasonably large.
*/
const SATURATION_THRESHOLD_PERCENT: usize = 10;

/**
A sentinel page that signals a hash covers too many pages.

Instead of storing all these pages, we just mark the value
as covering all pages.
*/
const SATURATED_PAGE_VALUE: u32 = u32::MAX;

/**
Calculate the index of a bucket to use.
*/
#[inline(always)]
const fn wrapping_bucket_index(index: u32, nbuckets: usize) -> usize {
    index as usize % nbuckets
}

/**
Calculate the maximum number of pages to store per hash
before treating the hash as saturated.
*/
#[inline(always)]
const fn max_page_values_per_hash(npages: usize) -> usize {
    let max = npages / SATURATION_THRESHOLD_PERCENT;

    if max == 0 {
        1
    } else {
        max
    }
}

/**
An in-memory builder for a `DiskMap4`.
*/
#[derive(Debug)]
pub struct DiskMap4Builder {
    entries: BTreeMap<u32, DiskMap4BuilderEntry>,
    npages: usize,
    max_page_values_per_hash: usize,
}

#[derive(Default, Debug)]
struct DiskMap4BuilderEntry {
    page_values: HashSet<u32>,
    saturated: bool,
}

impl DiskMap4Builder {
    /**
    Create a builder for a source with the given number of pages.
    */
    pub fn new(npages: usize) -> Self {
        assert_ne!(0, npages);

        DiskMap4Builder {
            entries: BTreeMap::new(),
            npages,
            max_page_values_per_hash: max_page_values_per_hash(npages),
        }
    }

    /**
    The number of pages in the source.
    */
    pub fn npages(&self) -> usize {
        self.npages
    }

    /**
    Insert the entries from `other` into `self`.
    */
    pub fn merge(&mut self, other: Self) {
        for (hash, entry) in other.entries {
            for page in entry.page_values {
                self.mark_hit(hash, page);
            }
        }
    }

    /**
    Mark a page as a hit for the value.
    */
    pub fn mark_hit(&mut self, hash_value: u32, page_value: u32) {
        let entry = self.entries.entry(hash_value).or_default();

        if !entry.saturated {
            // If the hash covers too many pages then just record it
            // as saturated. This stops the map growing too large for
            // degenerate inputs
            if max_page_values_per_hash(entry.page_values.len()) >= self.max_page_values_per_hash {
                entry.saturated = true;
                entry.page_values = Some(SATURATED_PAGE_VALUE).into_iter().collect();
            } else {
                entry.page_values.insert(page_value);
            }
        }
    }

    /**
    Serialize the builder into a byte buffer.
    */
    pub fn write_to_vec(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write(&mut buf).unwrap();
        buf
    }

    /**
    Serialize the builder into a writer, such as a file.
    */
    pub fn write(&self, mut into: impl Write) -> io::Result<()> {
        #[derive(Debug, Clone, Copy)]
        struct Key {
            hash_value: u32,
            pages_offset: u32,
            npage_values: usize,
        }

        // Determine a bucket size based on the amount of data to write
        let nentries = self.entries.len();
        let nbuckets = cmp::max(nentries / TARGET_BUCKET_SIZE_HASH_VALUES, 1);

        // Data is length-prefixed so buffer the buckets
        let mut key_buckets = vec![vec![]; nbuckets];
        let mut value_buckets = vec![vec![]; nbuckets];

        // Split the input into buckets
        for (hash_value, entry) in &self.entries {
            let bucket_index = wrapping_bucket_index(*hash_value, nbuckets);

            // Each hash bucket lists the hashes, which are unique, along with the number of pages it covers
            let key_bucket = &mut key_buckets[bucket_index];

            // Each page bucket stores the pages, which may be repeated
            let value_bucket = &mut value_buckets[bucket_index];

            key_bucket.push(Key {
                hash_value: *hash_value,
                pages_offset: u32::MAX,
                npage_values: entry.page_values.len(),
            });

            for page_value in &entry.page_values {
                value_bucket.push(*page_value);
            }
        }

        // The offset of the first bucket is at the end of the header
        // This process assigns offsets by scanning through
        // all the hash values and their page values
        let mut current_offset = nbuckets as u32 * 4;

        // Build the offsets while writing the header
        for key_bucket in &mut key_buckets {
            // Write the bucket offset
            into.write_all(&current_offset.to_le_bytes())?;

            // If the bucket is empty then use a sentinel bucket instead
            // that will always treat a hash as missing
            if key_bucket.len() == 0 {
                key_bucket.push(Key {
                    hash_value: 0,
                    pages_offset: u32::MAX,
                    npage_values: 0,
                });
            }

            current_offset += key_bucket.len() as u32 * 8;

            // The offsets for pages start right at the end of the hashes
            for Key {
                pages_offset,
                npage_values,
                ..
            } in key_bucket
            {
                *pages_offset = current_offset;
                current_offset += *npage_values as u32 * 4;
            }
        }

        // Write the buckets
        for (key_bucket, value_bucket) in key_buckets.iter().zip(value_buckets.iter()) {
            debug_assert_ne!(0, key_bucket.len());

            // Write the page offsets first
            for Key { pages_offset, .. } in key_bucket {
                debug_assert_ne!(0, *pages_offset);

                into.write_all(&pages_offset.to_le_bytes())?;
            }

            // Write the hashes next
            for Key { hash_value, .. } in key_bucket {
                into.write_all(&hash_value.to_le_bytes())?;
            }

            // Write the pages
            for page_value in value_bucket {
                into.write_all(&page_value.to_le_bytes())?;
            }
        }

        Ok(())
    }
}

/**
A disk-friendly immutable map of 32bit hashes to sets of 32bit page offsets.
*/
#[derive(Debug)]
pub struct DiskMap4<'a> {
    // Cache the number of buckets to avoid computing on every lookup
    nbuckets: usize,
    // The underlying map data
    map: &'a [u8],
}

impl<'a> DiskMap4<'a> {
    /**
    A map that treats any input as empty.
    */
    #[allow(dead_code)] // Keeping this function for future optimization
    pub const fn empty() -> Self {
        // A map that is guaranteed to always return empty for any input
        //
        // This map is not just zeroes. A typical kind of disk corruption
        // is getting a page full of zeroes, and we want to distinguish this from
        // a valid empty map
        const EMPTY_MAP: &'static [u8] = &[4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0];

        DiskMap4 {
            nbuckets: 1,
            map: EMPTY_MAP,
        }
    }

    /**
    A map that treats any input as saturated.
    */
    pub const fn saturated() -> Self {
        // A map that is guaranteed to always return saturated for any input
        //
        // This works by exercising corruption checking; the bucket offset
        // points outside the map
        const SATURATED_MAP: &'static [u8] = &[u8::MAX; 4];

        DiskMap4 {
            nbuckets: 1,
            map: SATURATED_MAP,
        }
    }

    /**
    Read a map from raw bytes.

    This method doesn't deserialize.
    */
    #[inline]
    pub const fn read(map: &'a [u8]) -> Self {
        if map.len() < 8 || map.len() % 4 != 0 {
            return read_corrupted();
        }

        // The offset to the first bucket marks the start of the data portion
        // of the file
        let buckets_offset = (u32::from_le_bytes([map[0], map[1], map[2], map[3]])) as usize;

        if buckets_offset < 4 || buckets_offset > map.len() {
            return read_corrupted();
        }

        let nbuckets = buckets_offset / 4;

        DiskMap4 { nbuckets, map }
    }

    /**
    Get a hitmap for a given value.
    */
    pub fn hits(&self, hash: u32) -> HitMap {
        let mut hits = HitMap::Unknown;
        self.lookup_hash::<true>(hash, &mut hits);

        // Catch differences between vectorized and non-vectorized
        // lookups in tests
        #[cfg(debug_assertions)]
        {
            let mut fallback = HitMap::Unknown;
            self.lookup_hash::<false>(hash, &mut fallback);

            assert_eq!(hits, fallback);
        }

        hits
    }

    /**
    Lookup a given value using a visitor.
    */
    pub fn lookup(&self, hash: u32, lookup: impl Lookup) {
        self.lookup_hash::<true>(hash, lookup)
    }

    #[inline(always)]
    fn lookup_hash<const SIMD: bool>(&self, hash_value: u32, mut lookup: impl Lookup) {
        let bucket_index = wrapping_bucket_index(hash_value, self.nbuckets);

        // Figure out what bucket we need to scan
        let bucket_start = unsafe { u32_unchecked(self.map, bucket_index * 4) as usize };

        // CORRUPTION: Invalid bucket length
        if bucket_start + 4 > self.map.len() {
            return lookup_corrupted(lookup);
        }

        let hash_values_end = unsafe { u32_unchecked(self.map, bucket_start) } as usize;

        // CORRUPTION: Invalid offset
        if hash_values_end > self.map.len() {
            return lookup_corrupted(lookup);
        }

        let hash_values_pages_offsets_len = hash_values_end.saturating_sub(bucket_start) / 2;

        // Offsets are written before hashes, so the hashes start halfway between the
        // start of the bucket and the start of the page offsets
        let hash_values_start = bucket_start + hash_values_pages_offsets_len;

        // Vectorized scan through the bucket looking for any hashes
        // that match the input
        let mut current_offset = prescan::<SIMD>(self.map, hash_value, hash_values_start, hash_values_end);

        debug_assert!(current_offset >= bucket_start);

        // Serially scan for hashes that match the input
        //
        // The vectorized scan should chew through most of the bucket
        // so if this loop runs it'll likely only be a few iterations
        while current_offset + 4 <= hash_values_end {
            let candidate_hash_value = unsafe { u32_unchecked(self.map, current_offset) };

            // If the hash is a hit then scan its pages
            if candidate_hash_value == hash_value {
                let page_values_start = unsafe { u32_unchecked(self.map, current_offset - hash_values_pages_offsets_len) as usize };

                // If there's a hash following this one then the pages to return
                // are the data up to the following hash's offset
                let page_values_end = if current_offset + 4 < hash_values_end {
                    unsafe { u32_unchecked(self.map, (current_offset + 4) - hash_values_pages_offsets_len) as usize }
                }
                // Otherwise the pages to return are the data to the end of the bucket
                else {
                    let bucket_end = {
                        let next_bucket_index = wrapping_bucket_index(bucket_index as u32 + 1, self.nbuckets);

                        // If the next bucket is wrapped then the end of
                        // the one we're interested in is the end of the file
                        if next_bucket_index != 0 {
                            unsafe { u32_unchecked(self.map, next_bucket_index * 4) as usize }
                        }
                        // Otherwise the start of the next bucket is the end
                        // of the one we're interested in
                        else {
                            self.map.len()
                        }
                    };

                    bucket_end
                };

                // CORRUPTION: Invalid offset
                if page_values_end > self.map.len() {
                    return lookup_corrupted(lookup);
                }

                current_offset = page_values_start;

                // Scan through the set of pages for this hash
                let mut hit = false;
                while current_offset + 4 <= page_values_end {
                    let page_value = unsafe { u32_unchecked(self.map, current_offset) };

                    if page_value == SATURATED_PAGE_VALUE {
                        lookup.saturated();
                        return;
                    } else {
                        hit = true;
                        lookup.hit(page_value);
                    }

                    current_offset += 4;
                }

                // If the set of pages was empty then return
                if !hit {
                    lookup.empty();
                }
                return;
            }

            current_offset += 4;
        }

        lookup.empty()
    }
}

#[cold]
fn lookup_corrupted(mut lookup: impl Lookup) {
    lookup.saturated();
}

#[cold]
const fn read_corrupted() -> DiskMap4<'static> {
    DiskMap4::saturated()
}

/**
Vectorized scanning for ARM.

Each iteration compares 4 hashes at a time.
*/
#[inline]
#[target_feature(enable = "neon")]
#[cfg(target_arch = "aarch64")]
unsafe fn prescan_neon(map: &[u8], hash: u32, mut offset: usize, end: usize) -> usize {
    use std::arch::aarch64::{vceqq_u32, vld1q_u32, vmaxvq_u32};

    let hash = vld1q_u32(&[hash, hash, hash, hash] as *const _);

    // Crunch through blocks of values until we reach one that contains
    // the hash we're looking for
    while offset + 16 <= end {
        let next_offset = offset + 16;

        let block = vld1q_u32(map.get_unchecked(offset..next_offset) as *const _ as *const u32);

        if vmaxvq_u32(vceqq_u32(hash, block)) != 0 {
            // If we match then return the position up to this block
            // and read its hits serially
            return offset;
        }

        offset = next_offset;
    }

    offset
}

/**
Vectorized scanning for x86 using AVX2.

Each iteration compares 8 hashes at a time.
*/
#[inline]
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn prescan_avx2(map: &[u8], hash: u32, mut offset: usize, end: usize) -> usize {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_setr_epi32};

    let hash = hash as i32;
    let hash = _mm256_setr_epi32(hash, hash, hash, hash, hash, hash, hash, hash);

    // Crunch through blocks of values until we reach one that contains
    // the hash we're looking for
    while offset + 32 <= end {
        let next_offset = offset + 32;

        let block = _mm256_loadu_si256(map.get_unchecked(offset..next_offset) as *const _ as *const __m256i);

        let candidate = _mm256_movemask_epi8(_mm256_cmpeq_epi32(hash, block));

        if candidate != 0 {
            // Blocks in AVX2 are large, so offset directly to the hit
            return offset + (candidate.trailing_zeros() as usize);
        }

        offset = next_offset;
    }

    offset
}

/**
A fast scan through the bucket that discards non-matching hashes.

This function can quickly determine whether a hash is present,
and return an offset close to it if it is.
*/
#[inline(always)]
fn prescan<const SIMD: bool>(map: &[u8], hash_value: u32, hash_values_start: usize, hash_values_end: usize) -> usize {
    if SIMD {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { prescan_neon(map, hash_value, hash_values_start, hash_values_end) };
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return unsafe { prescan_avx2(map, hash_value, hash_values_start, hash_values_end) };
            }
        }
    }

    let _ = (map, hash_value, hash_values_start, hash_values_end);

    hash_values_start
}

/**
A visitor for a specific value in a map that can be passed to `DiskMap4::lookup`.

Only one method on `Lookup` will be called by a given call to `DiskMap4::lookup`.
*/
pub trait Lookup {
    /**
    The value is not present in the map.

    This method will be called at most once.
    */
    fn empty(&mut self);

    /**
    The value has a hit for the given page.

    This method may be called multiple times.
    */
    fn hit(&mut self, page: u32);

    /**
    The value covers every page.

    This method will be called at most once.
    */
    fn saturated(&mut self);
}

impl<'a, V: Lookup + ?Sized> Lookup for &'a mut V {
    #[inline]
    fn empty(&mut self) {
        (**self).empty()
    }

    #[inline]
    fn hit(&mut self, page: u32) {
        (**self).hit(page)
    }

    #[inline]
    fn saturated(&mut self) {
        (**self).saturated()
    }
}

/**
A map of hits for a given value.
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HitMap {
    /**
    An unknown value.
    */
    Unknown,
    /**
    The value isn't present.
    */
    Empty,
    /**
    The value is present on every page.
    */
    Saturated,
    /**
    The value is present on a sparse set of pages.
    */
    Sparse(Vec<u32>),
}

impl Lookup for HitMap {
    #[inline]
    fn empty(&mut self) {
        *self = HitMap::Empty;
    }

    #[inline]
    fn saturated(&mut self) {
        *self = HitMap::Saturated;
    }

    #[inline]
    fn hit(&mut self, page: u32) {
        if let HitMap::Sparse(sparse) = self {
            sparse.push(page);
            sparse.sort();
        } else {
            *self = HitMap::Sparse(vec![page]);
        }
    }
}

#[inline(always)]
const unsafe fn u32_unchecked(data: &[u8], i: usize) -> u32 {
    #[cfg(debug_assertions)]
    {
        u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]])
    }
    #[cfg(not(debug_assertions))]
    {
        u32::from_le_bytes(*(data.as_ptr().offset(i as isize) as *const u8 as *const [u8; 4]))
    }
}
