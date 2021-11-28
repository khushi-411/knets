#pragma once

#include <iterator>
#include <iostreami.h>  // name was standardized
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

/* Reference: https://stackoverflow.com/questions/1120140
 */
class CSVRow
{
	public:
		std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] - (
